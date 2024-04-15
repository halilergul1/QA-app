# QA-app

## Directory Structure

This section provides an overview of the main directories and files in this repository, explaining how they contribute to the project.

### 📂  `Base Directory`
- **app.py**: The entry point of the program. Contains the main functionalities executed when the project is run via FAST api.
- **IndexManager.py**: Contains the IndexManager class with attributes and methods handling index loading, index initialization, creation, query engine creation and formatting output
- **ingest.py**: Implements the data ingestion part from pdfs into llamaindex nodes
- **index.py**: Takes the generated nodes and setup a Elasticsearch vector store and assemble a query engine of llamaindex.
- **querying.py**: Takes a query and generates a response.
- **evaluate.py**: Generates synthetic data out of document nodes for model evaluation via RelevancyEvaluator.
- **e2e_test.py**: contains methods for end-to-end test cases covering different possible scenarious while taking user queries etc.
- **config.py**: contains the configuration variables for the model.

### 📂 `/pdfs`
- **Description**: Includes data files (which are shared pdf files) used in the project.

### 📂 `/experiments`
- **Description**: Includes py notebooks that come with 2-3 different solutions for the QA task. The file semanticChunker.ipynb is the final solution method. The details were explained in experiments.pdf

### 📂 `/manual-test`
- **Description**: Contains index checker and loader files that implement the main pipeline with hardcoded variables.

### 📂 `/results`
- **Description**: Includes raw results of RelevancyEvaluator of the final method/solution (semanticChunker)

### 📄 `README.md`
- **Description**: Provides an overview of the project, installation instructions, and usage examples.

### 📄 `.gitignore`
- **Description**: Specifies intentionally untracked files to ignore.


## How to Setup the Service

This section details the steps required to set up and run the service locally for development, testing, and deployment using Docker. Follow these instructions to get your environment ready.

### Local Development Setup

1. **Create a Virtual Environment**: 
   - It's recommended to use a virtual environment to isolate package dependencies. To create a virtual environment, run:
     ```
     python3 -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```
       .\venv\Scripts\activate
       ```
     - On macOS and Linux:
       ```
       source venv/bin/activate
       ```

2. **Install Required Packages**:
   - Install all the necessary packages using pip. This includes libraries specified in your `requirements.txt`:
     ```
     pip install -r requirements.txt
     ```

3. **Install Test Dependencies**:
   - Ensure that your testing environment is also ready by installing the required packages for testing:
     ```
     pip install pytest
     ```

### Running the Application

To run the application using Docker, follow these steps:

1. **Set Your OPENAI_API_KEY**:
   - You have to set your OPENAI_API_KEY before running Docker as an environment variable. You can do this by running the following command in the terminal on macOS and Linux (You can use "set" instead of "export" on Windows):
     ```
     export OPENAI_API_KEY='your-api-key'
     ```

2. **Build the Docker Application**:
   - Use Docker Compose to build the application. This will read the `docker-compose.yml` file and set up the necessary Docker containers:
     ```
     docker-compose build
     ```

3. **Start the Application**:
   - Once the build is complete, start the application by running:
     ```
     docker-compose up
     ```
   - This command starts all the services defined in your Docker Compose configuration.

### Running Test Cases defined in e2e_test.py

To execute tests, first ensure that the Docker containers are up and running. Also make sure you properly followed the steps "Local Development Setup" above to create venv and install requirement files. You can perform the following:

1. **Start Docker Containers**:
   - If not already running, start your Docker containers:
     ```
     docker-compose up
     ```

2. **Run Tests**:
   - Execute the tests using `pytest` by running the following command in the terminal while you are in the project directory and the Docker containers are running:
     ```
     pytest
     ```

These steps will help you set up the development environment, run the application, and execute tests efficiently. Adjust the commands according to your specific configurations if necessary.