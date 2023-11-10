#!/bin/bash

required_python_version="python3.8"

check_python_version() {
    python_version=$($1 --version 2>&1)
    if [[ $python_version == Python\ 3.8* ]]; then
        return 0
    else
        return 1
    fi
}

if check_python_version "python3.8"; then
    python_executable="python3.8"
elif check_python_version "python3"; then
    python_executable="python3"
else
    echo "Python 3.8.x is not installed."
    return 1
fi

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Not in a virtual environment. Creating one..."

    $python_executable -m venv venv

    source venv/bin/activate
fi

pip install -r requirements.txt