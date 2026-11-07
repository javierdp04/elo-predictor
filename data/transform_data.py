"""Lichess PGN to CSV converter.

Generates two files:
  * <name>_games.csv -> metadata for each game (one row per game).
  * <name>_moves.csv -> moves with their evaluation (one row per move).

Only games with engine evaluation ([%eval ...]) are stored.
The rest are discarded without parsing their moves, which makes the
process much faster.

The main process reads the PGN and sends batches of evaluated games to a
pool of worker processes (one per CPU thread by default), which parse them
with python-chess and return the CSV text. Output order matches the PGN.

Usage:
    python transform_data.py [path.pgn] [--out-dir DIR] [--limit N]
                             [--workers N] [--batch-size N]
"""

import argparse
import csv
import io
import os
import re
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess.pgn
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_PGN = DATA_DIR / "lichess_db_standard_rated_2018-06.pgn"

# PGN headers -> metadata CSV column
HEADER_COLUMNS = {
    "Event": "event",
    "Site": "site",
    "White": "white",
    "Black": "black",
    "Result": "result",
    "UTCDate": "utc_date",
    "UTCTime": "utc_time",
    "WhiteElo": "white_elo",
    "BlackElo": "black_elo",
    "WhiteRatingDiff": "white_rating_diff",
    "BlackRatingDiff": "black_rating_diff",
    "WhiteTitle": "white_title",
    "BlackTitle": "black_title",
    "ECO": "eco",
    "Opening": "opening",
    "TimeControl": "time_control",
    "Termination": "termination",
}
GAME_FIELDS = ["game_id", *HEADER_COLUMNS.values(), "num_plies"]

MOVE_FIELDS = [
    "game_id",
    "ply",          # 1, 2, 3... (half-move)
    "move_number",  # move number as in the PGN
    "color",        # white / black
    "san",
    "uci",
    "eval",         # evaluation in pawns from White's point of view
    "mate",         # mate in N (positive: White wins, negative: Black wins)
    "clock",        # remaining seconds if the PGN includes [%clk]
    "fen",          # position after the move
]

HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')


def iter_raw_games(path: Path, progress: tqdm | None = None):
    """Split the PGN into (headers, movetext) blocks without parsing them.

    If a progress bar is given, it is advanced by the number of bytes read.
    """
    headers: dict[str, str] = {}
    movetext: list[str] = []
    pending_bytes = 0
    with open(path, "rb") as f:
        for raw_line in f:
            if progress is not None:
                pending_bytes += len(raw_line)
                if pending_bytes >= 1 << 20:  # update every ~1 MB
                    progress.update(pending_bytes)
                    pending_bytes = 0
            line = raw_line.decode("utf-8", errors="replace")
            if line.startswith("["):
                if movetext:
                    yield headers, "".join(movetext)
                    headers, movetext = {}, []
                m = HEADER_RE.match(line)
                if m:
                    headers[m.group(1)] = m.group(2)
            elif line.strip():
                movetext.append(line)
    if progress is not None and pending_bytes:
        progress.update(pending_bytes)
    if headers or movetext:
        yield headers, "".join(movetext)


def game_id_from_site(site: str) -> str:
    return site.rstrip("/").rsplit("/", 1)[-1] if site else ""


def extract_moves(game_id: str, movetext: str) -> list[list]:
    """Walk the moves with python-chess and return rows with their evaluation."""
    game = chess.pgn.read_game(io.StringIO(movetext))
    if game is None:
        return []

    rows = []
    board = game.board()
    for ply, node in enumerate(game.mainline(), start=1):
        move = node.move
        san = board.san(move)
        color = "white" if board.turn == chess.WHITE else "black"
        move_number = board.fullmove_number
        board.push(move)

        score = node.eval()
        eval_value, mate = "", ""
        if score is not None:
            white_score = score.white()
            if white_score.is_mate():
                mate = white_score.mate()
            else:
                eval_value = white_score.score() / 100

        clock = node.clock()
        rows.append([
            game_id,
            ply,
            move_number,
            color,
            san,
            move.uci(),
            eval_value,
            mate,
            "" if clock is None else clock,
            board.fen(),
        ])
    return rows


def process_batch(batch: list[tuple[dict, str]]):
    """Worker: parse a batch of games and return their CSV text.

    Returns a list with one (game_csv_line, moves_csv_text) per stored game,
    plus a list of error messages.
    """
    results = []
    errors = []
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")

    def take() -> str:
        text = buffer.getvalue()
        buffer.seek(0)
        buffer.truncate()
        return text

    for headers, movetext in batch:
        game_id = game_id_from_site(headers.get("Site", ""))
        try:
            moves = extract_moves(game_id, movetext)
        except Exception as exc:  # corrupt game: skip it
            errors.append(f"Error in game {game_id}: {exc}")
            continue
        if not moves:
            continue

        writer.writerow([
            game_id,
            *(headers.get(header, "") for header in HEADER_COLUMNS),
            len(moves),
        ])
        game_line = take()
        writer.writerows(moves)
        results.append((game_line, take()))
    return results, errors


def iter_batches(pgn_path: Path, progress: tqdm, batch_size: int, counters: dict):
    """Yield batches of games that contain [%eval]; count every game read."""
    batch = []
    for headers, movetext in iter_raw_games(pgn_path, progress):
        counters["read"] += 1
        # Fast filter: without [%eval] the game is neither parsed nor stored
        if "%eval" not in movetext:
            continue
        batch.append((headers, movetext))
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def convert(pgn_path: Path, out_dir: Path, limit: int | None = None,
            workers: int | None = None, batch_size: int = 200):
    workers = workers or os.cpu_count() or 1
    out_dir.mkdir(parents=True, exist_ok=True)
    games_csv = out_dir / f"{pgn_path.stem}_games.csv"
    moves_csv = out_dir / f"{pgn_path.stem}_moves.csv"

    counters = {"read": 0}
    kept = errors = 0
    start = time.time()
    progress = tqdm(total=pgn_path.stat().st_size, unit="B", unit_scale=True,
                    unit_divisor=1024, desc=pgn_path.name, dynamic_ncols=True)

    def update_postfix(refresh=False):
        progress.set_postfix_str(
            f"kept={kept:,} read={counters['read']:,} workers={workers}",
            refresh=refresh)

    with progress, \
         open(games_csv, "w", newline="", encoding="utf-8") as gf, \
         open(moves_csv, "w", newline="", encoding="utf-8") as mf, \
         ProcessPoolExecutor(max_workers=workers) as pool:
        csv.writer(gf, lineterminator="\n").writerow(GAME_FIELDS)
        csv.writer(mf, lineterminator="\n").writerow(MOVE_FIELDS)

        # Bounded queue of pending batches: keeps every worker busy without
        # reading the whole file into memory. Results are written in order.
        pending = deque()
        max_pending = workers * 4
        done = False

        def write_oldest():
            nonlocal kept, errors, done
            results, batch_errors = pending.popleft().result()
            for message in batch_errors:
                progress.write(message, file=sys.stderr)
            errors += len(batch_errors)
            for game_line, moves_text in results:
                gf.write(game_line)
                mf.write(moves_text)
                kept += 1
                if limit and kept >= limit:
                    done = True
                    break
            update_postfix()

        for batch in iter_batches(pgn_path, progress, batch_size, counters):
            pending.append(pool.submit(process_batch, batch))
            if len(pending) >= max_pending:
                write_oldest()
            if done:
                break
        while pending and not done:
            write_oldest()
        for future in pending:  # only left over when --limit was reached
            future.cancel()

        update_postfix(refresh=True)

    print(f"\nGames read:             {counters['read']}")
    print(f"Games with evaluation:  {kept}")
    print(f"Games with errors:      {errors}")
    print(f"Workers:                {workers}")
    print(f"Time:                   {time.time() - start:.0f}s")
    print(f"Metadata -> {games_csv}")
    print(f"Moves    -> {moves_csv}")


def main():
    parser = argparse.ArgumentParser(description="Convert a Lichess PGN to CSV.")
    parser.add_argument("pgn", nargs="?", type=Path, default=DEFAULT_PGN)
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of games with evaluation to store")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of worker processes (default: all CPU threads)")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Games sent to a worker at once (default: 200)")
    args = parser.parse_args()
    convert(args.pgn, args.out_dir, args.limit, args.workers, args.batch_size)


if __name__ == "__main__":
    main()
