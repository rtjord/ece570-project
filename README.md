# # CutMix Tiny Reproduction — Colab Instructions

This project is contained entirely in a single Google Colab notebook. No local setup is required.

## How to Run

1. **Open the notebook in Google Colab.**
   - Upload the `.ipynb` file to Colab, or open it from a shared Drive/GitHub link.

2. **Enable a GPU runtime.**
   - In Colab: `Runtime` → `Change runtime type`
   - Set:
     - **Hardware accelerator:** `GPU`
     - **GPU type:** **A100**
   - Click `Save`.

3. **Run the notebook top-to-bottom.**
   - Use `Runtime` → **Run all**,  
     **or** execute each cell manually **in order** from the top.
   - The notebook handles:
     - installing dependencies
     - downloading datasets
     - training models
     - saving checkpoints / results
     - producing plots and tables

4. **Do not skip cells.**
   - Later cells assume variables, models, and paths created earlier.

## Outputs

By the end of the notebook you should have:
- trained model checkpoints
- training/validation curves
- final Top-1 / Top-5 results for each configuration
- figures/tables used in the paper

## Notes / Tips

- The A100 is recommended for reasonable training time.
- If you re-run a later section, you may need to re-run earlier setup cells first.
- If Colab disconnects, re-open the notebook and run from the top again to restore state.
