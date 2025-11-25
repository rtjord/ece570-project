# # CutMix Tiny Reproduction — Colab Instructions

This project is contained entirely in a single Google Colab notebook. No local setup is required.

## How to Run

1. **Open the notebook in Google Colab.**
   - Upload the `.ipynb` file to Colab.

2. **Enable a GPU runtime.**
   - In Colab: `Runtime` → `Change runtime type`
   - Set:
     - **Hardware accelerator:** `A100 GPU`
     - **High-RAM**: `On`
   - Click `Save`.

3. **Run the notebook top-to-bottom.**
   - Use `Runtime` → **Run all**,  
     **or** execute each cell manually **in order** from the top.
   - The notebook handles:
     - downloading datasets
     - training models
     - saving checkpoints / results
     - producing plots and tables

4. **Do not skip cells.**
   - Later cells assume variables, models, and paths created earlier.

## Outputs

By the end of the notebook you should have:
- trained model checkpoints in the checkpoints/ directory
- training/validation curves for each configuration in cell outputs
- final Top-1 / Top-5 results for each configuration in cell outputs
- final results containing model, training history, and test results in results/ directory
- figures used in the paper in cell outputs and saved to root directory

## Notebook Organization

- All cells prior to Global Config define functions and classes.
- All cells below Global Config train and test a model or create figures.
- Editing the GLOBAL_CONFIG dict changes those settings across all models.
- Editing the CONFIG dict above each experiment only affects that particular experiment.


## Notes / Tips

- The A100 is recommended for reasonable training time.
- It could take several hours to train all models with the current configurations.
- You can reduce the number of epochs to reduce training time.
- If Colab disconnects, re-open the notebook and run from the top again to restore state.
