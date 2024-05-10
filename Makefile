install: 
	python -m pip install --upgrade pip &&\
		pip install -r src/Neural_Network_python/requirements.txt

lint: 
	pylint --disable=R,C src/Neural_Network_python/

black:
	python -m black src/Neural_Network_python/

ruff:
	ruff check ../src/
	ruff check --fix ../src/
	ruff format ../src/

cpplint:
	cpplint src/Simulation_C++/code/*.cpp