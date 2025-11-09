import tkinter as tk
from tkinter import ttk
import chess
import threading
import contextlib
import json
import pickle
from PIL import Image, ImageTk
import os
import sys
try:
    # When imported as a package
    from .play import pick_move, get_opening_move
except Exception:
    # When executed as a script (no package context), fall back to top-level import
    from play import pick_move, get_opening_move

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Bot")
        self.root.geometry("700x600")
        
        # Load the chess model and vocabulary
        # Model loading (pickle) and sklearn/joblib can print noisy messages; reduce logging
        import logging, contextlib, os
        logging.getLogger('joblib').setLevel(logging.ERROR)
        logging.getLogger('sklearn').setLevel(logging.ERROR)

        # Load vocab (best-effort)
        self.vocab = None
        self._vocab_load_error = None
        try:
            with open(os.path.join('data', 'vocab.json')) as f:
                self.vocab = json.load(f)
        except Exception as e:
            self.vocab = None
            self._vocab_load_error = str(e)

        # Load model while suppressing stdout/stderr to avoid flooding the terminal
        self.model = None
        self._model_load_error = None
        model_path = os.path.join('chess_forest_model.pkl')
        try:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                    try:
                        with open(model_path, 'rb') as f:
                            self.model = pickle.load(f)
                    except Exception as e:
                        # capture pickle/load errors
                        self._model_load_error = str(e)
        except FileNotFoundError:
            self._model_load_error = f"Model file not found: {model_path}"
        except Exception as e:
            if self._model_load_error is None:
                self._model_load_error = str(e)
        
        # Create a chess board
        self.board = chess.Board()
        self.move_history = []
        
        # Drag & selection state
        self.selected_square = None
        self.drag_text_id = None
        self.drag_symbol = None
        self.highlight_ids = []
        
        # Create the UI components
        self.create_widgets()
        
        # Update the board display
        self.update_display()
    
    def create_widgets(self):
        # Create a frame for the board
        self.board_frame = tk.Frame(self.root)
        self.board_frame.pack(pady=20)
        
        # Create a canvas for the chess board
        self.canvas = tk.Canvas(self.board_frame, width=400, height=400)
        self.canvas.pack()
        
        # Bind mouse events for pick-and-drop
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Create a frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)

        # Create input for user moves (kept for alternate entry)
        ttk.Label(control_frame, text="Your move:").grid(row=0, column=0, padx=5)
        self.move_entry = ttk.Entry(control_frame, width=10)
        self.move_entry.grid(row=0, column=1, padx=5)
        self.move_entry.bind('<Return>', self.on_move_submit)

        # Create a submit button
        submit_btn = ttk.Button(control_frame, text="Submit", command=self.on_move_submit)
        submit_btn.grid(row=0, column=2, padx=5)

        # Create a new game button
        new_game_btn = ttk.Button(control_frame, text="New Game", command=self.new_game)
        new_game_btn.grid(row=0, column=3, padx=5)

        # Create a status label
        self.status_var = tk.StringVar()
        self.status_var.set("Welcome to Chess Bot! You play as White.")
        status_lbl = ttk.Label(self.root, textvariable=self.status_var, font=('Arial', 12))
        status_lbl.pack(pady=10)
        # Surface any model/vocab load issues to the user
        if getattr(self, '_model_load_error', None):
            self.status_var.set(f"Warning: model load issue - {self._model_load_error}")
        elif getattr(self, '_vocab_load_error', None):
            self.status_var.set(f"Warning: vocab load issue - {self._vocab_load_error}")

        # Create a thinking indicator
        self.thinking_var = tk.StringVar()
        thinking_lbl = ttk.Label(self.root, textvariable=self.thinking_var, font=('Arial', 10))
        thinking_lbl.pack(pady=5)

        # Create a move history text box
        history_frame = tk.Frame(self.root)
        history_frame.pack(pady=10, fill=tk.X, padx=20)

        ttk.Label(history_frame, text="Move History:").pack(anchor=tk.W)

        # Add a vertical scrollbar and use a monospace font for alignment
        # Slightly larger size for better visibility
        self.history_text = tk.Text(history_frame, height=10, width=50, wrap=tk.WORD, font=('Courier', 11))
        self.history_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.history_text.config(state=tk.DISABLED)

        scroll = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_text['yscrollcommand'] = scroll.set

        # Populate history initially (empty)
        try:
            self.refresh_history_text()
        except Exception:
            # don't crash GUI creation if history render fails
            try:
                self.history_text.config(state=tk.DISABLED)
            except Exception:
                pass
    def update_display(self):
        # Clear canvas
        self.canvas.delete("all")
        self.highlight_ids = []

        # Draw squares and pieces
        for row in range(8):
            for col in range(8):
                x1, y1 = col * 50, row * 50
                x2, y2 = x1 + 50, y1 + 50
                color = "#FFFFFF" if (row + col) % 2 == 0 else "#86A666"
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")

                square = chess.square(col, 7 - row)
                piece = self.board.piece_at(square)
                if piece:
                    symbol = self.get_piece_symbol(piece)
                    self.canvas.create_text(x1 + 25, y1 + 25, text=symbol, font=('Arial', 24), tags=("piece", f"square_{square}"))

        # Draw rank numbers and file letters
        for i in range(8):
            # Ranks (1-8) on left
            self.canvas.create_text(5, i * 50 + 25, text=str(8 - i), anchor=tk.W)
            # Files (a-h) on bottom
            self.canvas.create_text(i * 50 + 25, 400 - 5, text=chr(97 + i), anchor=tk.S)

        # Update status
        if self.board.is_game_over():
            result = self.board.result()
            self.status_var.set(f"Game over! Result: {result}")
        else:
            self.status_var.set(f"{ 'White' if self.board.turn else 'Black' } to move")

    def refresh_history_text(self):
        """Rebuild the move history box from self.move_history using SAN and move numbers."""
        # Always operate with the text widget writable, then restore disabled state
        self.history_text.config(state=tk.NORMAL)
        try:
            self.history_text.delete(1.0, tk.END)

            if not self.move_history:
                # nothing to show
                return

            board = chess.Board()
            out_lines = []

            # Build SAN lines robustly: for each UCI in move_history, try to apply to a fresh board
            for idx, mv_uci in enumerate(self.move_history):
                san = mv_uci
                try:
                    # if mv_uci is already a Move object, coerce to string
                    if not isinstance(mv_uci, str):
                        mv_uci = str(mv_uci)

                    mv = chess.Move.from_uci(mv_uci)
                    # compute SAN safely
                    try:
                        san = board.san(mv)
                    except Exception:
                        san = mv_uci
                    # apply the move; use push_uci to be robust
                    try:
                        board.push(mv)
                    except Exception:
                        # fallback: try push_uci with the string
                        try:
                            board.push_uci(mv_uci)
                        except Exception:
                            # as a last resort, skip pushing to avoid breaking history building
                            pass
                except Exception:
                    # if parsing fails, try push_uci and use raw uci as SAN
                    try:
                        board.push_uci(mv_uci)
                    except Exception:
                        pass

                # format into move-numbered lines
                if idx % 2 == 0:
                    move_no = idx // 2 + 1
                    if idx == len(self.move_history) - 1:
                        out_lines.append(f"{move_no}. {san}")
                    else:
                        # start the line and wait for black reply
                        out_lines.append(f"{move_no}. {san}")
                else:
                    # append black move to previous line
                    # join last line with black SAN
                    if out_lines:
                        out_lines[-1] = out_lines[-1] + f" {san}"
                    else:
                        # fallback - shouldn't happen
                        out_lines.append(san)

            # Insert into text widget
            for l in out_lines:
                self.history_text.insert(tk.END, l + "\n")

            # scroll to end
            self.history_text.see(tk.END)
        finally:
            # always disable editing at the end
            try:
                self.history_text.config(state=tk.DISABLED)
            except Exception:
                pass
    
    def get_piece_symbol(self, piece):
        symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        return symbols[piece.symbol()]
    
    # Helper: convert canvas coordinates to chess square
    def square_at_coords(self, x, y):
        if x < 0 or y < 0 or x >= 400 or y >= 400:
            return None
        col = x // 50
        row = y // 50
        return chess.square(int(col), 7 - int(row))
    
    # Canvas event handlers for pick-and-drop
    # ...existing code...
    def on_canvas_click(self, event):
        sq = self.square_at_coords(event.x, event.y)
        if sq is None:
            return
        piece = self.board.piece_at(sq)
        # Only allow picking your side's pieces
        if piece and piece.color == self.board.turn:
            # list legal moves from this square
            legal_moves = [m for m in self.board.legal_moves if m.from_square == sq]
            if not legal_moves:
                self.status_var.set("Selected piece has no legal moves.")
                return

            # set selection and create floating drag symbol
            self.selected_square = sq
            self.drag_symbol = self.get_piece_symbol(piece)
            if self.drag_text_id:
                self.canvas.delete(self.drag_text_id)
            self.drag_text_id = self.canvas.create_text(event.x, event.y, text=self.drag_symbol, font=('Arial', 24), tags=("drag",))
            # highlight legal destinations
            self.show_legal_destinations(sq)
        else:
            self.selected_square = None

    def on_canvas_drag(self, event):
        # keep floating text following the mouse
        if self.drag_text_id:
            x = max(5, min(395, event.x))
            y = max(5, min(395, event.y))
            self.canvas.coords(self.drag_text_id, x, y)

    def on_canvas_release(self, event):
        # If nothing selected, just cleanup and return
        if self.selected_square is None:
            if self.drag_text_id:
                self.canvas.delete(self.drag_text_id)
                self.drag_text_id = None
            self.clear_highlights()
            return

        target_sq = self.square_at_coords(event.x, event.y)

        # remove floating drag symbol
        if self.drag_text_id:
            self.canvas.delete(self.drag_text_id)
            self.drag_text_id = None

        # cancel if released outside board or same square
        if target_sq is None or target_sq == self.selected_square:
            self.clear_highlights()
            self.selected_square = None
            return

        from_sq = self.selected_square
        piece = self.board.piece_at(from_sq)

        # Build move, handle promotion (default to queen)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(target_sq) in (0, 7):
            move = chess.Move(from_sq, target_sq, promotion=chess.QUEEN)
        else:
            move = chess.Move(from_sq, target_sq)

        # Execute move if legal
        if self.board.is_legal(move):
            self.board.push(move)
            self.move_history.append(move.uci())
            # Refresh the history display (SAN + numbering)
            self.refresh_history_text()
            try:
                self.root.update_idletasks()
            except Exception:
                pass
            self.update_display()

            # After player's move, if it's opponent's turn and game not over, let bot move
            if not self.board.is_game_over() and not self.board.turn:
                # only let bot play when it's Black's turn (we assume user is White)
                self.bot_make_move()
        else:
            try:
                self.status_var.set(f"Illegal move: {chess.square_name(from_sq)} → {chess.square_name(target_sq)}")
            except Exception:
                self.status_var.set("Illegal move, try again.")

        # Cleanup selection/highlights
        self.selected_square = None
        self.clear_highlights()
    
    def show_legal_destinations(self, from_sq):
        self.clear_highlights()
        for move in self.board.legal_moves:
            if move.from_square == from_sq:
                to_sq = move.to_square
                file = chess.square_file(to_sq)
                rank = chess.square_rank(to_sq)
                # convert to canvas coords
                col = file
                row = 7 - rank
                x1, y1 = col * 50, row * 50
                x2, y2 = x1 + 50, y1 + 50
                # semi-transparent oval to indicate allowed square
                hid = self.canvas.create_oval(x1+15, y1+15, x2-15, y2-15, fill="#f0e68c", outline="", stipple="gray50", tags=("highlight",))
                self.highlight_ids.append(hid)
    
    def clear_highlights(self):
        for hid in self.highlight_ids:
            try:
                self.canvas.delete(hid)
            except Exception:
                pass
        self.highlight_ids = []
    
    def on_move_submit(self, event=None):
        move_uci = self.move_entry.get().strip()
        self.move_entry.delete(0, tk.END)
        
        if move_uci.lower() == 'quit':
            self.root.quit()
            return
        
        try:
            move = chess.Move.from_uci(move_uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.move_history.append(move_uci)

                # Refresh history display with SAN and move numbers
                self.refresh_history_text()
                try:
                    self.root.update_idletasks()
                except Exception:
                    pass

                # Update the display
                self.update_display()
                
                # If game is not over, let the bot make a move
                if not self.board.is_game_over():
                    self.bot_make_move()
            else:
                self.status_var.set("Illegal move, try again.")
        except ValueError:
            self.status_var.set("Invalid move format, try again.")
    
    def bot_make_move(self):
        # Indicate that the bot is thinking
        self.thinking_var.set("Bot is thinking...")
        self.root.update()
        
        # Run the bot's move selection in a separate thread to keep the UI responsive
        def bot_thread():
            move_uci = None
            
            # First try opening book
            opening_move = get_opening_move(self.board, self.move_history)
            if opening_move:
                move_uci = opening_move
                thinking_info = "Using opening book"
            else:
                # Use minimax with appropriate depth based on game phase
                piece_count = len(self.board.piece_map())
                
                if piece_count > 25:  # Early game
                    depth = 3
                    max_time = 10
                    thinking_info = "Using lookahead (depth=3)"
                elif piece_count > 15:  # Mid game
                    depth = 4
                    max_time = 20
                    thinking_info = "Using lookahead (depth=4)"
                else:  # Late game
                    depth = 5
                    max_time = 25
                    thinking_info = "Using lookahead (depth=5)"
                
                # This runs the move selection without printing those annoying messages
                move_uci = pick_move(self.board, self.model, self.vocab, self.move_history)
            
            # Now update the UI from the main thread
            self.root.after(0, lambda: self.complete_bot_move(move_uci, thinking_info))
        
        # Start the bot's thinking thread
        thread = threading.Thread(target=bot_thread)
        thread.daemon = True
        thread.start()
    
    def complete_bot_move(self, move_uci, thinking_info):
        if move_uci:
            self.board.push_uci(move_uci)
            self.move_history.append(move_uci)

            # Optionally show how move was chosen in the status bar
            if thinking_info:
                self.status_var.set(thinking_info)

            # Refresh history display with SAN and move numbers
            self.refresh_history_text()
            try:
                self.root.update_idletasks()
            except Exception:
                pass

            # Clear thinking indicator
            self.thinking_var.set("")

            # Update the display
            self.update_display()
        else:
            self.thinking_var.set("")
            self.status_var.set("Bot couldn't find a legal move!")
    
    def new_game(self):
        self.board = chess.Board()
        self.move_history = []
        try:
            self.history_text.config(state=tk.NORMAL)
        except Exception:
            pass
        self.history_text.delete(1.0, tk.END)
        try:
            self.history_text.config(state=tk.DISABLED)
        except Exception:
            pass
        self.status_var.set("New game started. You play as White.")
        self.thinking_var.set("")
        self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()