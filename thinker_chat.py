# thinker-chat.py
import argparse
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
import sys
import time
import threading
import signal # To handle Ctrl+C gracefully with the spinner
import atexit # To ensure screen restoration on exit

DEFAULT_MODEL_PATH = "MoE-4bit"
DEFAULT_MAX_TOKENS = 16000 # Increased default max tokens
DEFAULT_TEMP = 0.6
DEFAULT_SEED = 0
REPLACEMENT_CHAR = "\ufffd" # Standard Unicode replacement character

# ANSI escape codes for alternate screen buffer
ENTER_ALT_SCREEN = "\x1b[?1049h"
EXIT_ALT_SCREEN = "\x1b[?1049l"
CLEAR_SCREEN = "\x1b[2J"
CURSOR_TO_HOME = "\x1b[H"

# Flag to track if alternate screen is active
alt_screen_active = False

def enter_alternate_screen():
    """Enters the alternate screen buffer and clears it."""
    global alt_screen_active
    if not alt_screen_active:
        # Enter alt screen, clear it, move cursor to top-left
        print(ENTER_ALT_SCREEN + CLEAR_SCREEN + CURSOR_TO_HOME, end="", flush=True)
        alt_screen_active = True

def exit_alternate_screen():
    """Exits the alternate screen buffer."""
    global alt_screen_active
    if alt_screen_active:
        print(EXIT_ALT_SCREEN, end="", flush=True)
        alt_screen_active = False

# Register exit_alternate_screen to be called automatically on script exit
# This handles normal exit, sys.exit(), Ctrl+C (after signal handler), and most errors
atexit.register(exit_alternate_screen)

class Spinner:
    """Simple CLI spinner running in a separate thread."""
    def __init__(self, message="Thinking..."):
        self.message = message
        # self.symbols = ['|', '/', '-', '\']
        self.symbols = ['◢', '◣', '◤', '◥'] # Alternative spinner characters
        self.running = False
        self.thread = None
        self.lock = threading.Lock() # Protect access to running flag

    def _spin(self):
        i = 0
        while True:
            with self.lock:
                if not self.running:
                    break
            # Write spinner frame
            # Use \r to return to beginning of line, works in alt screen
            sys.stdout.write(f"\r{self.message} {self.symbols[i % len(self.symbols)]}")
            sys.stdout.flush()
            time.sleep(0.15) # Spinner speed
            i += 1
        # Clear the spinner line upon stopping using \r
        sys.stdout.write('\r' + ' ' * (len(self.message) + 5) + '\r')
        sys.stdout.flush()

    def start(self):
        """Start the spinner in a background thread."""
        with self.lock:
            if not self.running:
                self.running = True
                self.thread = threading.Thread(target=self._spin, daemon=True)
                self.thread.start()

    def stop(self):
        """Stop the spinner and wait for the thread to finish."""
        needs_join = False
        with self.lock:
            if self.running:
                self.running = False
                needs_join = True
        if needs_join and self.thread:
            # Wait briefly for the thread to exit to avoid blocking
            self.thread.join(timeout=0.5)
            self.thread = None


# Global spinner instance to allow signal handler to stop it
spinner_instance = None

def signal_handler(sig, frame):
    """Handle Ctrl+C interruptions gracefully."""
    print("\nCaught interrupt, stopping generation...")
    if spinner_instance:
        spinner_instance.stop()
    # atexit handles screen restoration, just exit.
    sys.exit(0)

# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)


# --- ASCII Art Animation ---
THINKER_CHAT_ART = [
    "  _______ _    _ _____ _   _ _  ________ _____     _____ _    _       _______ ",
    " |__   __| |  | |_   _| \\ | | |/ /  ____|  __ \\   / ____| |  | |   /\\|__   __|",
    "    | |  | |__| | | | |  \| | ' /| |__  | |__) | | |    | |__| |  /  \  | |   ",
    "    | |  |  __  | | | | . ` |  < |  __| |  _  /  | |    |  __  | / /\ \ | |   ",
    "    | |  | |  | |_| |_| |\  | . \| |____| | \ \  | |____| |  | |/ ____ \| |   ",
    "    |_|  |_|  |_|_____|_| \_|_|\_\______|_|  \_\  \_____|_|  |_/_/    \_\_|   ",
    "                                                                              ",
    "                                                                              "
]

# Define the spinning characters for the wipe
WIPE_CHARS = ['|', '/', '-', '\\\\'] # Use double backslash for literal backslash

def animate_ascii_art(art_lines, delay=0.0075, wipe_chars=WIPE_CHARS): # Reduced default delay
    """Animates ASCII art by wiping it onto the screen with spinning characters."""
    max_len = max(len(line) for line in art_lines) if art_lines else 0
    height = len(art_lines)
    if height == 0 or not wipe_chars:
        return # Nothing to animate or no wipe characters

    # Wipe effect - diagonal wipe with spinning characters
    wipe_char_index = 0
    for i in range(max_len + height):
        sys.stdout.write(CURSOR_TO_HOME) # Go to top-left for redraw
        current_wipe_char = wipe_chars[wipe_char_index % len(wipe_chars)] # Cycle through wipe chars
        for r in range(height):
            line_content = ""
            current_line = art_lines[r]
            for c in range(max_len):
                # Calculate if the character at (r, c) should be revealed
                # Use a diagonal condition (r + c) < i
                if (c + r) < i:
                    if c < len(current_line):
                         line_content += current_line[c]
                    else:
                         line_content += " " # Pad shorter lines to max_len
                else:
                    # Draw the current wipe character if it's exactly on the boundary
                    if (c + r) == i and c < max_len:
                         line_content += current_wipe_char # Use the cycled character
                    else:
                         line_content += " " # Keep blank or padding

            # Print the line, ensuring it clears previous content up to max_len
            # Use print() instead of sys.stdout.write for automatic newline
            print(line_content.ljust(max_len), flush=False) # Delay flush until after full frame

        sys.stdout.flush() # Flush after drawing all lines for the frame
        time.sleep(delay)
        wipe_char_index += 1 # Move to next wipe character for the next frame

    # --- Clear the animation area using ANSI escape codes before final redraw ---
    sys.stdout.write(CURSOR_TO_HOME) # Go to top-left
    for r in range(height):
        # Move cursor to beginning of line 'r' (1-indexed)
        sys.stdout.write(f"[{r + 1};1H")
        # Clear the entire line
        sys.stdout.write("[2K")
    sys.stdout.flush() # Ensure clearing commands are sent
    # --- End clearing logic ---

    # Final redraw of the art without the wipe character
    sys.stdout.write(CURSOR_TO_HOME) # Ensure cursor is at home before drawing
    for line in art_lines:
        sys.stdout.write(line.ljust(max_len) + "\n")
    sys.stdout.flush()
    # Add a small pause after the animation completes
    time.sleep(0.3)
    # Print a separator line below the art
    print("-" * max_len)
    print() # Add an empty line for spacing


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Chat with an LLM using alternate screen, waiting for </think>.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MLX model directory (containing weights, tokenizer, config). "
             "Example: 'mlx-community/Mistral-7B-Instruct-v0.2-4bit-mlx'",
    )
    parser.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate per response.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=DEFAULT_TEMP,
        help="Sampling temperature. Higher values (e.g., 0.7) = more random, lower (e.g., 0.2) = more deterministic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for the pseudo-random number generator for reproducible results.",
    )
    return parser.parse_args()

def main_cli():
    global spinner_instance # Allow signal handler to access the spinner

    args = parse_args() # Parse arguments inside the function

    mx.random.seed(args.seed)
    print(f"Loading model from {args.model}...")
    try:
        model, tokenizer = load(args.model)
        print("Model loaded successfully.") # Confirmation before screen switch
    except Exception as e:
        print(f"\nError loading model: {e}")
        print("Please ensure the model path is correct and the model files exist.")
        sys.exit(1) # Exit before entering alt screen

    # --- Enter Alternate Screen ---
    # This happens only after successful model load
    enter_alternate_screen()

    # --- Animated Header ---
    animate_ascii_art(THINKER_CHAT_ART)

    # --- Print remaining info ---
    # print("=" * 10) # Replaced by animation
    # print(" Starting Thinker Chat ") # Replaced by animation
    # print("=" * 10) # Replaced by animation
    print("Enter 'q' or 'quit' to exit. Enter '/clear' to reset the chat.")
    print("Model:", args.model)
    print(f"Max Tokens: {args.max_tokens}, Temp: {args.temp}, Seed: {args.seed}")
    print("-" * 10)

    history = []

    while True:
        try:
            prompt_text = input(">> ")
            if prompt_text.lower() in ["q", "quit"]:
                break
            # --- Add /clear command ---
            if prompt_text.strip().lower() == "/clear":
                history.clear() # Clear the chat history
                # Clear screen, reset cursor, replay animation, print info
                sys.stdout.write(CLEAR_SCREEN + CURSOR_TO_HOME)
                sys.stdout.flush() # Ensure screen clears before animation
                animate_ascii_art(THINKER_CHAT_ART)
                print("Enter 'q' or 'quit' to exit.")
                print("Model:", args.model)
                print(f"Max Tokens: {args.max_tokens}, Temp: {args.temp}, Seed: {args.seed}")
                print("-" * 10)
                print() # Add spacing before next prompt
                continue # Skip the rest of the loop and ask for input again
            # --- End /clear command ---
            if not prompt_text.strip(): # Handle empty input
                continue
        except EOFError: # Handle Ctrl+D
            print("\nExiting.") # Print message inside alt screen
            break # Exit loop, atexit will restore screen

        history.append({"role": "user", "content": prompt_text})

        try:
            # Apply the chat template to format the history
            full_prompt = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True # Essential for instruction-tuned models
            )
        except Exception as e:
            print(f"\nError applying chat template: {e}")
            print("The model might lack a configured chat template. Skipping this turn.")
            history.pop() # Remove the user message that caused the error
            continue

        prompt = mx.array(tokenizer.encode(full_prompt))

        spinner = Spinner()
        spinner_instance = spinner # Make it globally accessible for signal handler
        spinner.start()

        state = "thinking" # Initial state: waiting for </think>
        buffer = "" # Accumulates text during thinking and newline stripping
        assistant_full_response_text = "" # Accumulate full text for history
        printed_something = False # Track if any output was actually printed

        try:
            # Use stream_generate which yields decoded text chunks
            sampler = make_sampler(temp=args.temp) # Use default sampler settings for now
            # Pass max_tokens directly to stream_generate
            generator = stream_generate(
                model, tokenizer, prompt, sampler=sampler, max_tokens=args.max_tokens
            )

            # Iterate through decoded text chunks from stream_generate
            # stream_generate handles the max_tokens limit internally.
            # Removed the manual chunk_count check.
            for response_chunk in generator:
                # response_chunk.text contains the decoded text string for this step
                # Replace potential decoding errors represented by REPLACEMENT_CHAR
                token_text = response_chunk.text.replace(REPLACEMENT_CHAR, "?") # Basic handling
                assistant_full_response_text += token_text # Append chunk to full response for history

                # --- State Machine for Response Handling (operates on decoded token_text chunks) ---
                if state == "thinking":
                    if token_text: # Only append if we got valid text
                        buffer += token_text
                    # Check if the closing tag is now present in the buffer
                    if "</think>" in buffer:
                        spinner.stop() # Found the tag, stop the spinner
                        spinner_instance = None

                        # Extract content appearing *after* the tag
                        think_tag_end_index = buffer.find("</think>") + len("</think>")
                        remainder = buffer[think_tag_end_index:]
                        buffer = remainder # Put remainder back in buffer for next state
                        state = "stripping_newlines"
                        # Process the remainder immediately in the next state block
                        token_text = "" # Clear token_text as its content is now in buffer for the next iteration

                # Use elif for state transitions
                elif state == "stripping_newlines":
                     # Add the current token's text to the buffer if not processed above
                     # The remainder from 'thinking' is already in buffer
                     if token_text: # Add any new text from this chunk
                          buffer += token_text

                     # Check buffer for leading newlines to remove
                     # This logic needs to handle partial chunks potentially splitting newlines
                     while buffer and state == "stripping_newlines": # Process buffer content
                         if buffer.startswith("\n\n"): # Check for double newline first
                             buffer = buffer[2:] # Remove double newline
                             # If something remains, print it and transition
                             if buffer:
                                 print(buffer, end="", flush=True)
                                 printed_something = True
                                 buffer = "" # Clear buffer after printing
                                 state = "streaming"
                             # If buffer became empty, stay in stripping, wait for more tokens
                         elif buffer.startswith("\n"): # Check for single newline
                             buffer = buffer[1:] # Remove single newline
                             if buffer:
                                 print(buffer, end="", flush=True)
                                 printed_something = True
                                 buffer = ""
                                 state = "streaming"
                         elif buffer: # Found non-newline content
                             print(buffer, end="", flush=True)
                             printed_something = True
                             buffer = ""
                             state = "streaming"
                         else: # Buffer is empty or only contained newlines that were stripped
                             break # Exit while loop, wait for next token_text

                     token_text = "" # Clear token_text as its content was processed or is empty

                elif state == "streaming":
                    # Buffer should be empty, just print the current token's text
                    if token_text:
                        print(token_text, end="", flush=True)
                        printed_something = True

            # --- End of generation loop ---
            spinner.stop() # Ensure spinner stops if max_tokens or EOS is reached (stream_generate handles EOS)
            spinner_instance = None

            # Use the accumulated text for history
            history.append({"role": "assistant", "content": assistant_full_response_text})

            if printed_something:
                print() # Add a final newline
            elif state == "thinking":
                 print("\n[Model stopped before </think> tag or generated empty response]")
            elif state == "stripping_newlines" and not buffer: # Check buffer state
                 print("\n[Model stopped after </think> tag, ended with newlines]")
            # If stopped while stripping and buffer has content (e.g. single \n), print it
            elif state == "stripping_newlines" and buffer:
                 print(buffer, end="\n") # Print remaining buffer content


        except Exception as e: # Catch errors during generation/streaming
            spinner.stop()
            spinner_instance = None
            print(f"\nAn error occurred during generation: {e}")
            # Add a placeholder to history indicating the error
            history.append({"role": "assistant", "content": "[Error during generation]"})

    # --- End of while True loop ---

    # This message will appear briefly in the alternate screen before atexit restores the original
    print("\nExiting chat.")

    # No finally: exit_alternate_screen() needed here, atexit handles it.
    # Any exception after enter_alternate_screen() but before atexit runs
    # will still trigger the atexit handler upon interpreter exit.


if __name__ == "__main__":
    main_cli() # Just call main_cli directly, it handles args now
    # Script ends here, atexit handler (exit_alternate_screen) is called automatically 