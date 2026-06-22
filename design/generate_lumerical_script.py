# design/generate_lumerical_script.py
import os
import pandas as pd

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")

def main():
    layout_csv = os.path.join(DATASET_DIR, "predicted_metalens_layout.csv")
    if not os.path.exists(layout_csv):
        raise FileNotFoundError("Run design.predict_layout first to generate the structural CSV.")

    df = pd.read_csv(layout_csv)
    grid_size_x = int(df["grid_x"].max()) + 1
    grid_size_y = int(df["grid_y"].max()) + 1
    grid_center_x = (grid_size_x - 1) / 2
    grid_center_y = (grid_size_y - 1) / 2
    
    # Unit cell dimensions
    unit_x = 500e-9  # 500 nm
    unit_y = 250e-9  # 250 nm
    height = 450e-9  # 450 nm

    # We use addrect; because it is compatible with both FDTD and MODE engines
    script_lines = [
        "clear;",
        "switchtolayout;",
        "selectall;",
        "deleteall;",
        "## Automatically add a fresh FDTD Simulation Region",
        "addfdtd;",
        "setnamed('FDTD','x min', -16e-6); setnamed('FDTD','x max', 16e-6);",
        "setnamed('FDTD','y min', -8e-6); setnamed('FDTD','y max', 8e-6);",
    ]

    print(f"Generating Lumerical Script for {len(df)} dimer structures...")
    
    for _, row in df.iterrows():
        gx, gy = row["grid_x"], row["grid_y"]
        pos_x = (gx - grid_center_x) * unit_x
        pos_y = (gy - grid_center_y) * unit_y

        # Pillar 1
        script_lines.append("addrect;")
        script_lines.append(f"set('name','Pillar_1_{gx}_{gy}');")
        script_lines.append(f"set('x',{pos_x - 125e-9}); set('y',{pos_y});")
        script_lines.append(f"set('z span',{height});")
        script_lines.append(f"set('x span',{row['x1_nm'] * 1e-9});")
        script_lines.append(f"set('y span',{row['y1_nm'] * 1e-9});")
        script_lines.append(f"set('z', 225e-9);")
        script_lines.append(f"set('material','Si (Silicon) - Palik');")
        script_lines.append(f"set('first axis','z'); set('rotation 1',{row['angle1_deg']});")

        # Pillar 2
        script_lines.append("addrect;")
        script_lines.append(f"set('name','Pillar_2_{gx}_{gy}');")
        script_lines.append(f"set('x',{pos_x + 125e-9}); set('y',{pos_y});")
        script_lines.append(f"set('z span',{height});")
        script_lines.append(f"set('x span',{row['x2_nm'] * 1e-9});")
        script_lines.append(f"set('y span',{row['y2_nm'] * 1e-9});")
        script_lines.append(f"set('z', 225e-9);")
        script_lines.append(f"set('material','Si (Silicon) - Palik');")
        script_lines.append(f"set('first axis','z'); set('rotation 1',{row['angle2_deg']});")

    output_lsf = os.path.join(OUTPUT_DIR, "construct_metalens.lsf")
    with open(output_lsf, "w", encoding="utf-8") as f:
        f.write("\n".join(script_lines))
        
    print(f"🎉 Lumerical Construction script saved to: {output_lsf}")
    print("👉 Open Lumerical GUI -> Open Script Editor -> Load this file and hit RUN!")

if __name__ == "__main__":
    main()