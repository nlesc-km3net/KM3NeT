.PHONY: test clean doc

all: test

test:
	nosetests -v

clean:
	find . -name 'core.*' -delete
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
