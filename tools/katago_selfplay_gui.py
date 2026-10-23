#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Watch a real KataGo GTP engine play both colors, game after game.
Python 3.8+, Tkinter (python3-tk on Ubuntu). No third-party pip packages.
"""
import argparse
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk

LETTERS = "ABCDEFGHJKLMNOPQRST"
STONE = {"B": 1, "W": 2}


class GTPError(RuntimeError):
    pass


class GTP:
    def __init__(self, executable, model, config, log_dir):
        cmd = [executable, "gtp", "-model", str(model)]
        if config:
            cmd.extend(["-config", str(config)])
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / "katago_stderr.log"
        self.log = open(self.log_path, "a", encoding="utf-8")
        self.log.write("\n--- KataGo started: %s ---\n" % datetime.now().isoformat())
        self.log.flush()
        try:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=self.log, bufsize=1,
                                         text=True, encoding="utf-8",
                                         errors="replace")
        except Exception:
            self.log.close()
            raise
        self.lock = threading.Lock()
        self.serial = 0

    def ask(self, command):
        with self.lock:
            self.serial += 1
            n = self.serial
            try:
                self.proc.stdin.write("%s %s\n" % (n, command))
                self.proc.stdin.flush()
            except (OSError, BrokenPipeError) as ex:
                raise GTPError("KataGo 没有正常启动。检查 %s: %s" % (self.log_path, ex))
            first = None
            # GTP response ends with a blank line, and starts with '=' or '?'.
            while True:
                line = self.proc.stdout.readline()
                if line == "":
                    raise GTPError("KataGo 已退出。请查看 %s" % self.log_path)
                line = line.rstrip("\r\n")
                if first is None:
                    if not line.startswith(("=", "?")):
                        continue
                    success = line[0] == "="
                    rem = line[1:].strip()
                    if rem.startswith(str(n)) and (len(rem) == len(str(n)) or rem[len(str(n))].isspace()):
                        rem = rem[len(str(n)):].strip()
                    first = [rem] if rem else []
                elif not line.strip():
                    result = "\n".join(first).strip()
                    if not success:
                        raise GTPError("GTP %s: %s" % (command, result))
                    return result
                else:
                    first.append(line)

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
        finally:
            self.log.close()


class GoBoard:
    """Local board for display; KataGo remains the authoritative rules engine."""
    def __init__(self, size):
        self.n = size
        self.grid = [[0 for _ in range(size)] for _ in range(size)]
        self.captured_by_black = 0
        self.captured_by_white = 0
        self.last = None

    def neighbors(self, x, y):
        for xx, yy in ((x-1, y), (x+1, y), (x, y-1), (x, y+1)):
            if 0 <= xx < self.n and 0 <= yy < self.n:
                yield (xx, yy)

    def group(self, x, y):
        color = self.grid[y][x]
        seen = {(x, y)}
        todo = [(x, y)]
        liberties = set()
        while todo:
            cx, cy = todo.pop()
            for nx, ny in self.neighbors(cx, cy):
                val = self.grid[ny][nx]
                if val == 0:
                    liberties.add((nx, ny))
                elif val == color and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    todo.append((nx, ny))
        return seen, liberties

    def play(self, color, move):
        mv = move.strip().upper()
        if mv == "PASS":
            self.last = None
            return
        if mv == "RESIGN":
            return
        m = re.fullmatch(r"([A-HJ-T])(\d{1,2})", mv)
        if not m:
            raise ValueError("落子坐标格式不对: %s" % move)
        x = LETTERS.index(m.group(1))
        y = self.n - int(m.group(2))
        if x >= self.n or not 0 <= y < self.n or self.grid[y][x] != 0:
            raise ValueError("棋盘落子不合法: %s" % move)
        who = STONE[color]
        self.grid[y][x] = who
        removed = 0
        for nx, ny in self.neighbors(x, y):
            if self.grid[ny][nx] != 0 and self.grid[ny][nx] != who:
                stones, liberties = self.group(nx, ny)
                if not liberties:
                    for ax, ay in stones:
                        self.grid[ay][ax] = 0
                    removed += len(stones)
        own, own_liberties = self.group(x, y)
        if not own_liberties:
            raise ValueError("出现自杀着法，棋盘可能和引擎规则不一致: %s" % move)
        if who == 1:
            self.captured_by_black += removed
        else:
            self.captured_by_white += removed
        self.last = (x, y)


def sgf_coord(move, size):
    if move.lower() == "pass":
        return ""
    m = re.fullmatch(r"([A-HJ-T])(\d{1,2})", move.upper())
    if not m:
        return ""
    x = LETTERS.index(m.group(1))
    y = size - int(m.group(2))
    if not 0 <= x < size or not 0 <= y < size:
        return ""
    return chr(97 + x) + chr(97 + y)


def save_sgf(directory, number, size, komi, moves, result):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    filename = root / ("KataGo_%dx%d_%s_%03d.sgf" % (
        size, size, datetime.now().strftime("%Y%m%d_%H%M%S"), number))
    result = re.sub(r"[\\\]\[]", "", result)
    header = "(;GM[1]FF[4]CA[UTF-8]SZ[%d]KM[%s]RU[Chinese]PB[KataGo]PW[KataGo]RE[%s]" % (
        size, komi, result)
    with filename.open("w", encoding="utf-8") as f:
        f.write(header)
        for color, move in moves:
            if move.upper() != "RESIGN":
                f.write(";%s[%s]" % (color, sgf_coord(move, size)))
        f.write(")\n")
    return filename


class App:
    def __init__(self, args):
        self.args = args
        self.root = tk.Tk()
        self.root.title("KataGo 自我对弈 · 自动连续下棋")
        self.root.geometry("1020x760")
        self.root.minsize(760, 630)
        self.root.configure(bg="#f1f5ef")
        self.events = queue.Queue()
        self.stop_flag = threading.Event()
        self.pause_flag = threading.Event()
        self.next_flag = threading.Event()
        self.engine = None
        self.delay = args.delay
        self.game = 0
        self.board = GoBoard(args.size)
        self.move_count = 0
        self.turn = "B"
        self.result = ""
        self.move_log = []
        self.drawn = False
        self.setup_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.worker_thread = threading.Thread(target=self.worker, daemon=True)
        self.worker_thread.start()
        self.root.after(80, self.drain_events)

    def setup_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Jade.TButton", background="#dceae0", foreground="#174a35",
                        padding=(10, 9), font=("Sans", 11))
        style.map("Jade.TButton", background=[("active", "#c5ddcc")])
        top = tk.Frame(self.root, bg="#1e5039", height=70)
        top.pack(side="top", fill="x")
        tk.Label(top, text="KataGo × KataGo", bg="#1e5039", fg="white",
                 font=("Sans", 20, "bold")).pack(side="left", padx=23, pady=13)
        self.game_label = tk.Label(top, text="正在启动引擎…", bg="#1e5039",
                                   fg="#dcece1", font=("Sans", 12))
        self.game_label.pack(side="right", padx=24)

        content = tk.Frame(self.root, bg="#f1f5ef")
        content.pack(fill="both", expand=True, padx=12, pady=10)
        left = tk.Frame(content, bg="#f1f5ef")
        left.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(left, width=620, height=620, bg="#dcbb85",
                                highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _: self.redraw())

        panel = tk.Frame(content, width=290, bg="#f1f5ef")
        panel.pack(side="right", fill="y", padx=(13, 0))
        panel.pack_propagate(False)
        self.state_label = tk.Label(panel, text="加载模型中…", wraplength=265,
                                    anchor="w", justify="left", bg="#f1f5ef",
                                    fg="#194b36", font=("Sans", 13, "bold"))
        self.state_label.pack(fill="x", pady=(6, 12))
        self.move_label = tk.Label(panel, text="第 0 手", anchor="w",
                                   bg="#f1f5ef", font=("Sans", 12))
        self.move_label.pack(fill="x", pady=4)
        self.capture_label = tk.Label(panel, text="黑提 0 子 · 白提 0 子",
                                      anchor="w", bg="#f1f5ef", font=("Sans", 11))
        self.capture_label.pack(fill="x", pady=4)
        self.pause_button = ttk.Button(panel, text="暂停", style="Jade.TButton",
                                        command=self.toggle_pause)
        self.pause_button.pack(fill="x", pady=(18, 6))
        ttk.Button(panel, text="直接开始下一局", style="Jade.TButton",
                   command=self.next_game).pack(fill="x", pady=5)
        tk.Label(panel, text="每步间隔（秒）", bg="#f1f5ef", anchor="w").pack(
            fill="x", pady=(16, 3))
        self.speed_label = tk.Label(panel, text="%.1f 秒" % self.delay,
                                    bg="#f1f5ef", fg="#335d45")
        self.speed_label.pack(anchor="e")
        ttk.Scale(panel, from_=0.0, to=4.0, value=self.delay,
                  command=self.update_delay).pack(fill="x", pady=(2, 12))
        tk.Label(panel, text="落子记录", bg="#f1f5ef", fg="#194b36",
                 anchor="w", font=("Sans", 11, "bold")).pack(fill="x", pady=(5, 6))
        frame = tk.Frame(panel, bg="#f1f5ef")
        frame.pack(fill="both", expand=True, pady=(0, 7))
        self.history = tk.Text(frame, width=26, font=("DejaVu Sans Mono", 10),
                               bg="#fffdf8", fg="#304034", relief="flat",
                               state="disabled", padx=9, pady=8)
        self.history.pack(side="left", fill="both", expand=True)
        bar = ttk.Scrollbar(frame, command=self.history.yview)
        bar.pack(side="right", fill="y")
        self.history.configure(yscrollcommand=bar.set)
        tk.Label(panel, text="双停结束 · 自动续局 · SGF 自动保存",
                 bg="#f1f5ef", fg="#66796c", font=("Sans", 9),
                 wraplength=260).pack(fill="x", pady=(6, 4))
        self.redraw()

    def update_delay(self, raw):
        self.delay = round(float(raw), 1)
        self.speed_label.configure(text="%.1f 秒" % self.delay)

    def toggle_pause(self):
        if self.pause_flag.is_set():
            self.pause_flag.clear()
            self.pause_button.configure(text="暂停")
        else:
            self.pause_flag.set()
            self.pause_button.configure(text="继续")

    def next_game(self):
        self.next_flag.set()
        self.pause_flag.clear()
        self.pause_button.configure(text="暂停")

    def sleep_control(self, duration):
        remaining = max(0.0, duration)
        while remaining > 0.001 and not self.stop_flag.is_set() and not self.next_flag.is_set():
            if self.pause_flag.is_set():
                time.sleep(0.08)
                continue
            step = min(remaining, 0.05)
            time.sleep(step)
            remaining -= step
        return not (self.stop_flag.is_set() or self.next_flag.is_set())

    def ready(self):
        while self.pause_flag.is_set() and not self.stop_flag.is_set() and not self.next_flag.is_set():
            time.sleep(0.08)
        return not self.stop_flag.is_set() and not self.next_flag.is_set()

    def worker(self):
        try:
            self.engine = GTP(self.args.engine, self.args.model,
                              self.args.config, self.args.records)
            engine = self.engine
            engine.ask("boardsize %d" % self.args.size)
            engine.ask("kata-set-rules chinese")
            engine.ask("komi %s" % self.args.komi)
            engine.ask("kata-set-param maxVisits %d" % self.args.visits)
            self.events.put(("ready",))
            game = 0
            while not self.stop_flag.is_set():
                self.next_flag.clear()
                game += 1
                engine.ask("clear_board")
                moves = []
                passes = 0
                color = "B"
                self.events.put(("new", game))
                finished = False
                result = ""
                while not self.stop_flag.is_set() and not self.next_flag.is_set():
                    if not self.ready():
                        break
                    move = engine.ask("genmove %s" % ("black" if color == "B" else "white")).strip().splitlines()[0].strip().upper()
                    if self.next_flag.is_set() or self.stop_flag.is_set():
                        break
                    if move != "RESIGN":
                        moves.append((color, move))
                    self.events.put(("move", color, move, len(moves)))
                    if move == "RESIGN":
                        result = ("W" if color == "B" else "B") + "+R"
                        finished = True
                        break
                    if move == "PASS":
                        passes += 1
                    else:
                        passes = 0
                    if passes >= 2:
                        try:
                            result = engine.ask("final_score").strip()
                        except GTPError:
                            result = "?"
                        finished = True
                        break
                    color = "W" if color == "B" else "B"
                    if not self.sleep_control(self.delay):
                        break
                if self.stop_flag.is_set():
                    break
                if self.next_flag.is_set():
                    continue
                if finished:
                    path = save_sgf(self.args.records, game, self.args.size,
                                    self.args.komi, moves, result)
                    self.events.put(("end", result, str(path)))
                    if not self.sleep_control(self.args.between_games):
                        continue
        except Exception as ex:
            self.events.put(("error", "%s: %s" % (type(ex).__name__, ex)))
        finally:
            if self.engine:
                try:
                    self.engine.close()
                except Exception:
                    pass

    def drain_events(self):
        try:
            for _ in range(100):
                try:
                    event = self.events.get_nowait()
                except queue.Empty:
                    break
                typ = event[0]
                if typ == "ready":
                    self.state_label.configure(text="模型已加载，开始下棋")
                elif typ == "new":
                    self.game = event[1]
                    self.board = GoBoard(self.args.size)
                    self.move_count = 0
                    self.turn = "B"
                    self.game_label.configure(text="第 %d 局 · %d×%d · %d visits" % (
                        self.game, self.args.size, self.args.size, self.args.visits))
                    self.state_label.configure(text="● 黑棋思考中")
                    self.move_label.configure(text="第 0 手")
                    self.capture_label.configure(text="黑提 0 子 · 白提 0 子")
                    self.history.configure(state="normal")
                    self.history.delete("1.0", "end")
                    self.history.configure(state="disabled")
                    self.redraw()
                elif typ == "move":
                    _, color, move, number = event
                    try:
                        self.board.play(color, move)
                    except ValueError as ex:
                        self.state_label.configure(text=str(ex))
                    self.move_count = number
                    self.turn = "W" if color == "B" else "B"
                    if move == "RESIGN":
                        self.state_label.configure(text=("黑" if color == "B" else "白") + "棋认输")
                    elif move == "PASS":
                        self.state_label.configure(text=("黑" if color == "B" else "白") + "棋停一手")
                    else:
                        self.state_label.configure(text=("黑" if color == "B" else "白") + "棋落子 " + move)
                    self.move_label.configure(text="第 %d 手 · %s" % (number, move))
                    self.capture_label.configure(text="黑提 %d 子 · 白提 %d 子" % (
                        self.board.captured_by_black, self.board.captured_by_white))
                    self.history.configure(state="normal")
                    self.history.insert("end", "%d. %s  %s\n" % (
                        number, "黑" if color == "B" else "白", move))
                    self.history.see("end")
                    self.history.configure(state="disabled")
                    self.redraw()
                elif typ == "end":
                    self.result = event[1]
                    self.state_label.configure(text="本局结束 · %s\n棋谱保存：%s" % (
                        self.result, os.path.basename(event[2])))
                elif typ == "error":
                    self.state_label.configure(text="错误：\n" + event[1])
                    self.game_label.configure(text="引擎已停止")
        finally:
            if not self.stop_flag.is_set():
                self.root.after(80, self.drain_events)

    def redraw(self):
        canvas = self.canvas
        canvas.delete("all")
        n = self.args.size
        width = max(canvas.winfo_width(), 440)
        height = max(canvas.winfo_height(), 440)
        margin = 41
        pitch = min((width-2*margin)/(n-1), (height-2*margin)/(n-1))
        bx = (width-pitch*(n-1))/2
        by = (height-pitch*(n-1))/2
        x0, y0 = bx, by
        x1, y1 = bx+pitch*(n-1), by+pitch*(n-1)
        canvas.create_rectangle(x0-pitch*.48, y0-pitch*.48,
                                x1+pitch*.48, y1+pitch*.48,
                                outline="#cda467", width=2)
        for i in range(n):
            px = bx + i*pitch
            py = by + i*pitch
            canvas.create_line(px, y0, px, y1, fill="#71542e", width=1)
            canvas.create_line(x0, py, x1, py, fill="#71542e", width=1)
            canvas.create_text(px, y1+pitch*.34, text=LETTERS[i],
                               fill="#6f512e", font=("Sans", max(8, int(pitch*.22))))
            canvas.create_text(x0-pitch*.34, py, text=str(n-i),
                               fill="#6f512e", font=("Sans", max(8, int(pitch*.22))))
        dots = {9:[2,4,6], 11:[2,5,8], 13:[3,6,9],
                15:[3,7,11], 19:[3,9,15]}.get(n, [])
        for x in dots:
            for y in dots:
                px, py = bx+x*pitch, by+y*pitch
                rr = max(2.1, pitch*.065)
                canvas.create_oval(px-rr, py-rr, px+rr, py+rr,
                                   fill="#6e5330", outline="")
        radius = pitch*.46
        for y in range(n):
            for x in range(n):
                color = self.board.grid[y][x]
                if color == 0:
                    continue
                px, py = bx+x*pitch, by+y*pitch
                canvas.create_oval(px-radius+1.4, py-radius+2.1,
                                   px+radius+1.4, py+radius+2.1,
                                   fill="#9d8159", outline="")
                canvas.create_oval(px-radius, py-radius, px+radius, py+radius,
                                   fill="#202522" if color==1 else "#fafaf3",
                                   outline="#111915" if color==1 else "#c3c5bd",
                                   width=1.5)
                rr = radius*.13
                canvas.create_oval(px-radius*.52, py-radius*.59,
                                   px-radius*.52+rr*2, py-radius*.59+rr*2,
                                   fill="#626863" if color==1 else "#ffffff", outline="")
        if self.board.last:
            x, y = self.board.last
            px, py = bx+x*pitch, by+y*pitch
            r = max(3.6, radius*.23)
            canvas.create_oval(px-r, py-r, px+r, py+r,
                               fill="#c4422c", outline="#f8e4d0", width=1)

    def close(self):
        self.stop_flag.set()
        if self.engine:
            try:
                if self.engine.proc.poll() is None:
                    self.engine.proc.terminate()
            except OSError:
                pass
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    p = argparse.ArgumentParser(description="KataGo real self-play GUI, continuous matches")
    p.add_argument("--engine", default="katago", help="KataGo executable; default: katago on PATH")
    p.add_argument("--model", required=True, help="KataGo native neural net .bin(.gz)")
    p.add_argument("--config", default=None, help="KataGo GTP config .cfg (needed unless default config exists)")
    p.add_argument("--size", type=int, choices=[9, 11, 13, 15, 19], default=9)
    p.add_argument("--visits", type=int, default=16, help="Search visits per move (default 16)")
    p.add_argument("--komi", type=float, default=7.5)
    p.add_argument("--delay", type=float, default=0.7, help="Delay between moves, seconds")
    p.add_argument("--between-games", type=float, default=3.0)
    p.add_argument("--records", default="selfplay_records", help="SGF and KataGo stderr output directory")
    args = p.parse_args()
    if args.visits < 1 or args.delay < 0 or args.between_games < 0:
        p.error("visits must be positive; delays cannot be negative")
    executable = shutil.which(args.engine) if not os.path.isfile(args.engine) else args.engine
    if not executable:
        p.error("找不到 KataGo 可执行程序：%s。请用 --engine 指定 katago 的路径。" % args.engine)
    args.engine = executable
    if not Path(args.model).is_file():
        p.error("找不到模型文件: %s" % args.model)
    args.model = str(Path(args.model).resolve())
    if args.config:
        if not Path(args.config).is_file():
            p.error("找不到配置文件: %s" % args.config)
        args.config = str(Path(args.config).resolve())
    App(args).run()


if __name__ == "__main__":
    main()