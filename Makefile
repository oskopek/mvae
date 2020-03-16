.PHONY = clean check test conda download_data

clean:
	yapf -r -i mt/ tests/

check: clean
	flake8 .

test: check
	pytest 2>&1

conda:
	# Make sure to install miniconda first.
	conda update conda
	conda env create -f environment.yml

download_data:
	pip install --no-deps geoopt==0.1.0
	python -m data.download
