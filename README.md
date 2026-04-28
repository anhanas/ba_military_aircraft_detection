# Military Aircraft Detection Data Preparation

This project prepares the `military-aircraft-recognition-dataset` for training an Oriented Bounding Box (OBB) model (such as YOLO).

## Setup Environment

To ensure reproducibility across different machines without committing heavy environments to GitHub, we use a virtual environment (`venv`) combined with a `requirements.txt` file. The virtual environment directory is ignored in `.gitignore`.

Follow these steps to get started:

### 1. Create a Virtual Environment
Run the following command to create a virtual environment named `.venv` in the project root:
```bash
python -m venv .venv
```

### 2. Activate the Environment
Activate the environment before installing dependencies or running the code.

**On macOS / Linux:**
```bash
source .venv/bin/activate
```

**On Windows:**
```cmd
.venv\Scripts\activate
```

### 3. Install Dependencies
Install the required packages (`tqdm`, `jupyter`, etc.) by running:
```bash
pip install -r requirements.txt
```

### 4. Run the Preparation Notebook
We do not track the raw dataset or the generated YOLO labels in git to prevent repository bloat. Instead, the notebook will automatically download the raw dataset directly from Kaggle for you.

Run all the cells. It will automatically download the `.zip` file using curl, extract it, split the data, normalize the bounding boxes, and organize everything in `data/train`, `data/validation`, and `data/test`!
*(Note: You must have access to download this Kaggle dataset via curl, or authenticate if required.)*

### 5. Verify with Visualization
Once the dataset is prepared, you can visualize the results.
Run the notebook to randomly pick images from your generated `train` folder, convert the YOLO OBB labels back to pixel coordinates, and display the original image side-by-side with its bounding box overlay. This ensures your data preparation pipeline worked flawlessly!
