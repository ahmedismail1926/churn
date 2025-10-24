"""
Model Training with Output Logging
Runs the full training pipeline and saves output to a file
"""
import sys
from pathlib import Path
from model_training import main

# Redirect output to file
output_file = Path("model_training_output.txt")

class Tee:
    """Write to both file and stdout"""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()

if __name__ == "__main__":
    with open(output_file, 'w', encoding='utf-8') as f:
        # Redirect stdout and stderr to both console and file
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        sys.stdout = Tee(sys.stdout, f)
        sys.stderr = Tee(sys.stderr, f)
        
        try:
            print(f"Output will be saved to: {output_file.absolute()}")
            print("="*80)
            main()
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        print(f"\n✓ Training complete! Output saved to: {output_file.absolute()}")
