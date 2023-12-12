# Visual Search Engine API

The Visual Search Engine API is designed to provide image similarity search functionality. Given an input image, the API compares it to a pre-trained model and returns the top 10 visually similar images along with additional information.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the Flask App](#running-the-flask-app)
  - [Endpoint](#endpoint)
  - [Testing with Postman](#testing-with-postman)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [License](#license)

## Getting Started

### Prerequisites

- Python 3.x
- pip (Python package installer)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/visual-search-engine-api.git
   cd visual-search-engine-api
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On Unix or MacOS:

     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Flask App

```bash
python run.py
```

The Flask app will start on `http://localhost:5000`.

### Endpoint

- **Endpoint:** `/api/predict`
- **Method:** `POST`
- **Parameters:**
  - `image` (file): The image file to process (required, mutually exclusive with `url`).
  - `url` (string): The URL of the image to process (required, mutually exclusive with `image`).

### Testing with Postman

You can use Postman to test the API. Make a `POST` request to `http://localhost:5000/api/predict` with either the `image` file or `url` parameter.

## Project Structure

The project follows the following structure:

- `api/`: Contains the Flask application.
  - `__init__.py`: Initialization file for the Flask app.
  - `model_service.py`: Module handling model-related logic.
  - `routes.py`: Module defining API routes.
- `data/`: Directory for storing data files.
- `models/`: Directory for storing model-related files.
-`temp/`: Directory for storing images uploaded
- `tests/`: Directory for testing scripts.
- `venv/`: Virtual environment directory.
- `run.py`: Script to run the Flask app.

## Documentation

For detailed documentation on API usage, refer to the Swagger documentation provided in the `/API` directory.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```