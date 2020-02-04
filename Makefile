.PHONY = clean check types test conda download_data genplots genlatentspace

clean:
	yapf -r -i mt/ tests/

check: clean
	flake8 .

types: check
	mypy -p mt 2>&1
	mypy -p tests 2>&1

test: check #types
	pytest 2>&1

conda:
	# Make sure to install miniconda first.
	conda update conda
	conda env create -f environment.yml

download_data:
	pip install --no-deps git+https://github.com/geoopt/geoopt.git
	python -m data.download

genplots:
	python -m mt.visualization.generate_plots --glob 'lsf.*' --plot 'models'

genlatentspace:
	python -m mt.visualization.latent_space --path 'path_to_chkpt_dir'
