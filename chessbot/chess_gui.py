import tkinter as tk
from tkinter import ttk
import chess
import subprocess
import threading
import re
import json
import pickle
from PIL import Image, ImageTk
import os
import sys
from play import pick_move, get_opening_move

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess Bot")
        self.root.geometry("700x600")
        
        # Load the chess model and vocabulary
        with open("data/vocab.json") as f:
            self.vocab = json.load(f)
        
        with open("chess_forest_model.pkl", 'rb') as f:
            self.model = pickle.load(f)
        
        # Create a chess board
        self.board = chess.Board()
        self.move_history = []
        
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
        
        # Create a frame for controls
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        # Create input for user moves
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
        
        # Create a thinking indicator
        self.thinking_var = tk.StringVar()
        thinking_lbl = ttk.Label(self.root, textvariable=self.thinking_var, font=('Arial', 10))
        thinking_lbl.pack(pady=5)
        
        # Create a move history text box
        history_frame = tk.Frame(self.root)
        history_frame.pack(pady=10, fill=tk.X, padx=20)
        
        ttk.Label(history_frame, text="Move History:").pack(anchor=tk.W)
        
        self.history_text = tk.Text(history_frame, height=5, width=40)
        self.history_text.pack(fill=tk.X)
    
    def update_display(self):
        # Clear the canvas
        self.canvas.delete("all")
        
        # Draw the chess board
        for row in range(8):
            for col in range(8):
                x1, y1 = col * 50, row * 50
                x2, y2 = x1 + 50, y1 + 50
                
                # Determine square color
                if (row + col) % 2 == 0:
                    color = "#FFFFFF"  # White
                else:
                    color = "#86A666"  # Green
                
                # Draw the square
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black")
                
                # Get the piece at this position
                square = chess.square(col, 7 - row)  # Chess uses different coordinate system
                piece = self.board.piece_at(square)
                
                if piece:
                    # Create a text representation of the piece
                    symbol = self.get_piece_symbol(piece)
                    self.canvas.create_text(x1 + 25, y1 + 25, text=symbol, font=('Arial', 24))
        
        # Draw rank numbers and file letters
        for i in range(8):
            # Ranks (1-8)
            self.canvas.create_text(5, i * 50 + 25, text=str(8 - i), anchor=tk.W)
            # Files (a-h)
            self.canvas.create_text(i * 50 + 25, 400 - 5, text=chr(97 + i), anchor=tk.S)
        
        # Update status
        if self.board.is_game_over():
            result = self.board.result()
            self.status_var.set(f"Game over! Result: {result}")
        else:
            self.status_var.set(f"{'White' if self.board.turn else 'Black'} to move")
    
    def get_piece_symbol(self, piece):
        symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        return symbols[piece.symbol()]
    
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
                
                # Update history text
                self.history_text.insert(tk.END, f"White: {move_uci}\n")
                self.history_text.see(tk.END)
                
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
            
            # Update history text
            self.history_text.insert(tk.END, f"{thinking_info}\nBlack: {move_uci}\n")
            self.history_text.see(tk.END)
            
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
        self.history_text.delete(1.0, tk.END)
        self.status_var.set("New game started. You play as White.")
        self.thinking_var.set("")
        self.update_display()

if __name__ == "__main__":
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()