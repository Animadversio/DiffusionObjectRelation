#!/usr/bin/env python3
"""
Simple script to run evaluation for a single model
"""

import subprocess
import sys
import os

def run_single_evaluation(model_name, text_encoder_type=None):
    """Run evaluation for a single model"""
    
    print(f"🚀 Running evaluation for: {model_name}")
    
    # Build command
    cmd = [
        "python", "experimental_scripts/posthoc_generation_train_traj_eval_cli.py",
        "--model_run_name", model_name
    ]
    
    if text_encoder_type:
        cmd.extend(["--text_encoder_type", text_encoder_type])
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the evaluation
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2 hour timeout
        
        if result.returncode == 0:
            print(f"✅ Successfully evaluated {model_name}")
            print("Output preview:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print(f"❌ Failed to evaluate {model_name}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ Evaluation timed out for {model_name}")
        return False
    except Exception as e:
        print(f"💥 Exception for {model_name}: {e}")
        return False

def main():
    """Main function"""
    
    if len(sys.argv) < 2:
        print("Usage: python run_single_model_eval.py <model_name> [text_encoder_type]")
        print("\nAvailable models:")
        print("  objrel_T5_DiT_B_pilot")
        print("  objrel_T5_DiT_mini_pilot")
        print("  objrel_rndembdposemb_DiT_B_pilot")
        print("  objrel_rndembdposemb_DiT_micro_pilot")
        print("  objrel_rndembdposemb_DiT_nano_pilot")
        print("  objrel_rndembdposemb_DiT_mini_pilot")
        print("  objrel_rndemb_DiT_B_pilot")
        print("  objrel_T5_DiT_B_pilot_WDecay")
        print("  objrel_T5_DiT_mini_pilot_WDecay")
        return
    
    model_name = sys.argv[1]
    text_encoder_type = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = run_single_evaluation(model_name, text_encoder_type)
    
    if success:
        print(f"\n🎉 Evaluation completed successfully for {model_name}")
    else:
        print(f"\n💥 Evaluation failed for {model_name}")

if __name__ == "__main__":
    main() 