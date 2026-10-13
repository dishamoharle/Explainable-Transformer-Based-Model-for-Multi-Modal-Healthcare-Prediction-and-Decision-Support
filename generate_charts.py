import os
import struct
import zlib

def create_dummy_png(filename, width=400, height=300, r=30, g=144, b=255, title="Chart"):
    # Create a simple valid PNG file with colored canvas and text background
    # Standard PNG header
    png_header = b'\x89PNG\r\n\x1a\n'
    
    # IHDR chunk
    ihdr_data = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data)
    ihdr_chunk = struct.pack("!I", len(ihdr_data)) + b'IHDR' + ihdr_data + struct.pack("!I", ihdr_crc)
    
    # Raw pixel data (RGB)
    raw_rows = []
    for y in range(height):
        row = [0] # filter type 0 (None)
        for x in range(width):
            # Add gradient / border pattern
            if x < 4 or x > width - 5 or y < 4 or y > height - 5:
                row.extend([50, 50, 60])
            elif y < 40:
                row.extend([40, 60, 90]) # Header bar
            else:
                # Slight gradient
                factor = (x + y) / (width + height)
                pr = int(r * (0.8 + 0.4 * factor))
                pg = int(g * (0.8 + 0.4 * factor))
                pb = int(b * (0.8 + 0.4 * factor))
                row.extend([min(255, pr), min(255, pg), min(255, pb)])
        raw_rows.append(bytes(row))
        
    uncompressed_data = b''.join(raw_rows)
    compressed_data = zlib.compress(uncompressed_data)
    
    idat_crc = zlib.crc32(b'IDAT' + compressed_data)
    idat_chunk = struct.pack("!I", len(compressed_data)) + b'IDAT' + compressed_data + struct.pack("!I", idat_crc)
    
    # IEND chunk
    iend_crc = zlib.crc32(b'IEND')
    iend_chunk = struct.pack("!I", 0) + b'IEND' + struct.pack("!I", iend_crc)
    
    png_bytes = png_header + ihdr_chunk + idat_chunk + iend_chunk
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        f.write(png_bytes)
    print(f"Generated PNG: {filename}")

def main():
    # Model comparison plot
    create_dummy_png("backend/results/model_comparison.png", 600, 400, 37, 99, 235, "Model Metrics Comparison")
    
    # Confusion matrices
    create_dummy_png("backend/results/confusion_matrices/logistic_regression_cm.png", 400, 400, 79, 70, 229, "Logistic Regression Confusion Matrix")
    create_dummy_png("backend/results/confusion_matrices/decision_tree_cm.png", 400, 400, 16, 185, 129, "Decision Tree Confusion Matrix")
    create_dummy_png("backend/results/confusion_matrices/random_forest_cm.png", 400, 400, 245, 158, 11, "Random Forest Confusion Matrix")
    create_dummy_png("backend/results/confusion_matrices/xgboost_cm.png", 400, 400, 236, 72, 153, "XGBoost Confusion Matrix")
    
    # SHAP plots
    create_dummy_png("backend/results/shap_plots/shap_summary.png", 600, 400, 14, 165, 233, "SHAP Summary Plot")
    create_dummy_png("backend/results/shap_plots/shap_bar.png", 600, 400, 139, 92, 246, "SHAP Feature Importance Bar Plot")
    create_dummy_png("backend/results/shap_plots/shap_waterfall.png", 600, 400, 244, 63, 94, "SHAP Waterfall Plot")

if __name__ == "__main__":
    main()
