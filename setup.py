import setuptools

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the contents of your requirements file
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name="thinker-chat", # Replace with your desired package name
    version="0.1.0", # Initial version
    author="Your Name", # Replace with your name
    author_email="your.email@example.com", # Replace with your email
    description="A simple CLI chat interface using MLX models with think/response handling.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/thinker-chat", # Replace with your project's URL
    # packages=setuptools.find_packages(), # Commented out: Use py_modules for single script
    # If thinker-chat.py remains a standalone script, use py_modules instead:
    py_modules=["thinker_chat"], # Use py_modules for the single script
    install_requires=requirements, # Reads dependencies from requirements.txt
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Choose an appropriate license
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
    ],
    python_requires='>=3.8', # Specify your minimum Python version
    entry_points={
        'console_scripts': [
            'thinker-chat=thinker_chat:main_cli', # Creates the 'thinker-chat' command that runs main_cli
        ],
    },
) 