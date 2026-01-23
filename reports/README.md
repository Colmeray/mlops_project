# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

`![my_image](figures/<image>.<extension>)`

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

or

```bash
uv add typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to either fill out the `requirements.txt`/`requirements_dev.txt` files or keeping your
    `pyproject.toml`/`uv.lock` up-to-date with whatever dependencies that you are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [x] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [x] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [ ] Consider running a hyperparameter optimization sweep (M14)
* [ ] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [x] Add a linting step to your continuous integration (M17)
* [x] Add pre-commit hooks to your version control setup (M18)
* [x] Add a continues workflow that triggers when data changes (M19)
* [x] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [ ] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [ ] Create a FastAPI application that can do inference using your model (M22)
* [ ] Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [ ] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Setup collection of input-output data from your deployed application (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Create an architectural diagram over your MLOps pipeline
* [ ] Make sure all group members have an understanding about all parts of the project
* [ ] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

68

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s245465, s245509, s245647, s245243

### Question 3
> **Did you end up using any open-source frameworks/packages not covered in the course during your project? If so**
> **which did you use and how did they help you complete the project?**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

Yes, we used two open-source packages that were not covered in the course: kagglehub and Pillow.

We used kagglehub to automatically download and cache the dataset from Kaggle, which made the data acquisition step fully reproducible and removed the need for manual downloads.

We used Pillow (PIL) to load and process images before converting them to tensors for training. It allowed us to easily open, convert, and validate image files in different formats.

All other libraries used in the project (such as PyTorch and Typer) were already part of the course framework.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We managed dependencies with uv, which provides fast and reproducible dependency resolution. All project dependencies and their version constraints are defined in pyproject.toml, including the required Python version. From this file, uv generates a lockfile (uv.lock) that pins exact package versions and resolved wheels, ensuring the environment is fully reproducible across machines.

The development workflow is centered around this lockfile: we commit both pyproject.toml and uv.lock to version control, so every team member uses the same dependency graph. When dependencies change, we update pyproject.toml and regenerate the lockfile with uv lock, followed by uv sync to apply the changes.

For a new member to get an exact copy of the environment they would need to do the following:
- Install uv and a compatible python version
- Clone the repository
- Run uv sync, which automatically creates a virtual environment and installs all dependencies exactly as specified in uv.lock

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

--- question 5 fill here ---

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

Yes, we used tools to keep the code clean and consistent.
We used Ruff for both linting and formatting, which helped us automatically find mistakes and keep the same coding style in all files. We also ran these checks using GitHub Actions, so every change is tested before being merged.We used Python type hints in several parts of the project to make it clear what kind of data each function expects and returns.

These concepts are important in larger projects because many people may work on the same code. If the code is messy or unclear, it becomes hard to understand and easy to break. Linting, typing, and documentation help make the code easier to read, safer to change, and simpler to maintain.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

In total we implemented 6 tests.

One test checks the data pipeline: preprocess creates the expected output files (classes.json and all.csv) and that MyDataset can load images and return tensors with the correct shape and label type.

Four tests check the models: both VGG16Transfer and SimpleModel produce outputs with the correct shape, and the VGG16 transfer setup correctly freezes the feature extractor while still allowing gradients on the classifier head.

Finally, one test is a training smoke test that runs one batch and verifies the training loop returns valid metrics.

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

The total code coverage of our project is 82%, which means that most of our source code is executed when running the test suite. This shows that the most important parts of the system, such as the data pipeline, models, and training logic, are well tested.

However, even if we had 100% coverage, we would not trust the code to be completely error free. Coverage only measures whether a line of code was executed, not whether the result was correct or whether all edge cases were tested. Logical mistakes, unexpected inputs, and performance issues can still exist even when every line is covered by a test.

Code coverage is therefore a useful indicator of how well the project is tested, but it should not be seen as proof that the system is perfect. Reliable software also requires meaningful test cases, good design, and continuous validation.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

Yes, our workflow made use of both branches and pull requests to structure development and maintain code quality. We followed a branch-based workflow where the main branch represented the stable, working version of the project. Each team member worked on their own feature branch (for example, for data processing or model development), which allowed parallel development without interfering with others’ work.

When a feature was ready, it was merged into main through a pull request. Pull requests were used to review changes, discuss design decisions, and catch potential bugs before they were integrated. This review step helped ensure that new code was consistent with the existing codebase and did not break the pipeline.

Using branches and pull requests also gave us a clear project history and made it easy to revert changes if something went wrong. Overall, this workflow improved collaboration, reduced merge conflicts, and increased the reliability of the codebase.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We did not use DVC in our project. However, using DVC would have been beneficial for managing and versioning our dataset and processed data. For example, our project relied on a large image dataset downloaded from Kaggle and later processed into structured metadata (classes.json) and an index file (all.csv).

With DVC, we could have tracked different versions of the raw dataset, preprocessing outputs, and even trained model checkpoints in a reproducible way. This would make it easier for team members to retrieve the exact same data state, compare different preprocessing pipelines, and roll back to earlier versions if something went wrong.

Additionally, DVC would allow us to store large files outside Git while still keeping lightweight pointers under version control, improving collaboration, storage efficiency, and experiment reproducibility.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We use GitHub Actions for continuous integration in our project. Our CI setup consists of several automated workflows that run whenever changes are pushed to the main branch or when a pull request is opened.

We have a unit testing workflow that runs our tests using pytest on multiple operating systems (Ubuntu, macOS, and Windows) and on multiple Python versions (3.11 and 3.12). This ensures that our code works consistently across different environments. The workflow also uses caching to speed up dependency installation.

We also have a linting workflow based on ruff, which checks code formatting and quality on every push and pull request. This helps us catch errors early and keep the code readable and consistent.

In addition, we run a pre-commit workflow that automatically executes all hooks defined in .pre-commit-config.yaml before code can be merged. This enforces basic quality checks such as removing trailing whitespace and validating YAML files.

To keep our dependencies up to date, we use Dependabot, which automatically creates pull requests when new versions of Python packages are available. We also have a scheduled workflow that updates our pre-commit hooks weekly.

Overall, our CI setup helps prevent bugs, maintain code quality, and keep our project secure and reproducible.

Example workflow: https://github.com/Colmeray/mlops_project/actions/workflows/tests.yaml

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We configured our experiments using hydra config files which made it easy to manage hyperparameters and run reproducible runs without hardcoding values in train.py or such. The default settings live in configs/config.yaml and are passed into the training script as a cfg object. For new experiments, we simply override parameters from the CLI for example:

"python -m project.train model.name=vgg16 lr=3e-4 batch_size=64 wandb.enable=false"

This lets us quickly change batch size, learning rate, model choice and toggle Weights & Biases logging while keeping the training code clean and consistent.


### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We ensured reproducibility by combining Hydra configs fixed random seeds and same environments. Each run loads a default config.yaml and any CLI overrides are applied explicitly, so the exact hyperparameters are always known. Hydra also saves the full resolved config for every run in the output folder, making it easy to rerun the same setup later. We set a global seed (for Python/NumPy/PyTorch) to make data splits and training as deterministic as possible. In addition, we used Weights & Biases to log the final config, metrics, and artifacts (e.g., model checkpoints), so results are traceable. Finally, we relied on locked dependencies (uv.lock) and Docker to keep the software environment consistent across machines.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

--- question 14 fill here ---

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

We used Docker to create a reproducible environment for training our machine learning models. Instead of relying on local Python installations and system-specific dependencies, we packaged the entire project, including the code, configuration files, and all dependencies, into a Docker image. This ensured that our experiments could be run consistently on different machines.

Our main Docker image is a training container. When the container starts, it automatically downloads the dataset using KaggleHub, preprocesses the images, and runs the training script. This allowed us to validate that the full pipeline works from a clean state.

To build the image, we run:
"docker build -t mlops-train -f dockerfiles/train.dockerfile ."

To run the container:
"docker run --rm -v $(pwd)/data:/app/data mlops-train"

This mounts the local data directory so that preprocessing results can be reused.

Link to Dockerfile: dockerfiles/train.dockerfile

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

When we encountered bugs during development, we mainly used a combination of print statements and running the code in small steps to identify where errors occurred. We also relied on pytest tests and pre-commit hooks (Ruff, Black, and mypy) to catch syntax errors, formatting problems, and simple logic mistakes early. When dependency or environment issues occurred, we reproduced the problem inside our Docker container to ensure that the error was not caused by local setup differences.

We also experimented with profiling using PyTorch’s built-in profiler. By running a short profiling session during training, we could inspect which parts of the training loop were most expensive. This helped us verify that the forward pass and backpropagation dominated runtime, which is expected for deep learning models.

We do not consider the code perfect, but the combination of debugging, testing, and profiling helped us improve stability and performance.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

--- question 17 fill here ---

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

--- question 18 fill here ---

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

--- question 19 fill here ---

![buckets overview from one user](reports/figures/buckets_jo.png)


### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

--- question 20 fill here ---

### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:



### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We managed to train it in cloud engine but only partially as we had some problems with getting the updates to weights and biases

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:

We did not implement a production API for our model as part of this project. However, if we were to deploy the model as a service, we would expose it through a simple FastAPI application.

The API would load the trained model at startup and provide an endpoint for example, "/predict" that accepts an image file or JSON input. The input would be preprocessed in the same way as during training, and the model would return a predicted class label and confidence score.

To make the API more robust, we would add input validation, error handling, and request logging. We would also include a health-check endpoint such as "/health" so that monitoring tools could verify that the service is running. Finally, we would containerize the API using Docker so it could be deployed consistently on any platform or cloud environment.

This approach would allow the model to be used by other systems in a simple and scalable way.

### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We did not deploy an API for this project. The main reason is that the focus of the project was on building and validating a reproducible machine learning pipeline, including data preprocessing, model training, testing, and continuous integration, rather than on serving the model as a web service.

Deploying the model as an API would require building and maintaining a separate service layer, handling user requests, and ensuring that the model runs reliably in a production setting. This was part of the assignment, but we did not manage to complete this step within the project.

However, the way our project is structured means that an API and deployment could be added later without major changes to the code. The model training and inference logic are separated from the rest of the system, which makes it easier to extend the project with a deployment step in the future.

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

We did not perform unit testing or load testing of an API, because we did not finish implementing the API for our model. However, if the API had been completed, we would have tested it using the same approach described in the course.

For functionality testing, we would use FastAPI’s TestClient together with pytest to simulate API calls and verify that the endpoints return the correct status codes and responses for different inputs. These tests would be placed in a separate integrationtests folder to clearly differentiate them from normal unit tests.

For load testing, we would use the Locust framework to simulate many users sending requests to the API at the same time. This would allow us to measure response time and throughput, and observe how the API behaves under increasing load. Running Locust in headless mode would also make it possible to integrate these tests.

Even though we did not implement this step, the project could easily be extended with API and performance testing using this setup.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not implement monitoring of a deployed model, because the model was not exposed as a running service. However, monitoring would be an important part of a real production system and would help ensure that the model continues to perform well over time.

With monitoring in place, we could track metrics such as prediction frequency, response time, and error rates to detect performance issues early. We could also monitor changes in the input data distribution and compare them with the training data to detect data drift. If the model’s predictions or confidence scores started to change significantly, this could indicate that the model is no longer well suited to new data.

Monitoring would therefore help maintain the reliability and accuracy of the system, and allow us to retrain or update the model when necessary, making sure of the long-term stability of the application.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

--- question 27 fill here ---

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

Nothing new was implemented.

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

--- question 29 fill here ---

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The biggest challenge for us was definitely the cloud setup (GCP). We spent almost three full project days working only on it, and it quickly became a repeated pattern: things that worked locally did not work in the cloud environment. Even worse, when we managed to fix one issue, it often introduced new problems somewhere else in the pipeline. Overall, cloud deployment became a major obstacle because it forced us to spend a lot of time debugging environment issues instead of focusing on model development.

The reason for this was mainly that the cloud environment behaved differently from our local setup, even when using the same code and configuration. For example, our Dockerfile worked reliably on our local machines together with Weights & Biases, but on GCP it sometimes failed without a clear explanation. In some cases, the container would build successfully but crash during execution or fail to connect properly. We never found one single root cause, but it likely involved differences in permissions, networking, container runtime behavior, or authentication configuration. Eventually, after multiple attempts and adjustments, we managed to get it working in the cloud as well.

In addition, we tried to rent a GPU instance to make the workflow scalable and more realistic, but this introduced new problems. We struggled to configure our uv and pyproject.toml dependencies to support GPU-enabled installations. Small version mismatches between CUDA-related packages and the PyTorch ecosystem caused installation failures or runtime errors. This highlighted how fragile ML environments can be when switching between CPU and GPU setups, especially in the cloud.


### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

--- question 31 fill here ---
