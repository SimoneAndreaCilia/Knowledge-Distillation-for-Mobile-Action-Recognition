import torch
import pandas as pd
from src.evaluation.profile import profile_model
from src.models import build_model

def main():
    print("Initializing Teacher (ResNet-50)...")
    teacher = build_model("teacher", num_classes=51, pretrained=False)
    
    print("Initializing Student (MobileNet3D width 1.0)...")
    student = build_model("student", num_classes=51, width_mult=1.0)
    
    print("Initializing Student (MobileNet3D width 1.5)...")
    student_15 = build_model("student", num_classes=51, width_mult=1.5)
    
    print("\nProfiling Teacher... (This might take a few seconds on CPU)")
    teacher_results = profile_model(teacher, device=torch.device("cpu"), measure_latency_flag=True)
    teacher_results["Model"] = "Teacher (ResNet3D-50)"
    
    print("\nProfiling Student 1.0... (This might take a few seconds on CPU)")
    student_results = profile_model(student, device=torch.device("cpu"), measure_latency_flag=True)
    student_results["Model"] = "Student (MobileNet3D width 1.0)"

    print("\nProfiling Student 1.5... (This might take a few seconds on CPU)")
    student15_results = profile_model(student_15, device=torch.device("cpu"), measure_latency_flag=True)
    student15_results["Model"] = "Student (MobileNet3D width 1.5)"
    
    results = [teacher_results, student_results, student15_results]
    
    # Save to CSV
    csv_file = "profiling_results.csv"
    df = pd.DataFrame(results)
    
    # Reorder columns to have Model first
    cols = ['Model'] + [c for c in df.columns if c != 'Model']
    df = df[cols]
    
    df.to_csv(csv_file, index=False)
    print(f"\n✅ Results saved to {csv_file}\n")
    
    # Print markdown table for easy copy-paste
    print("Here is the Markdown table for your REPORT.md:")
    print("-" * 50)
    print(df.to_markdown(index=False))
    print("-" * 50)

if __name__ == "__main__":
    main()
