.PHONY: test clean doc


all: test

test:
	nosetests -v

doc:
	$(eval DIFF := $(shell git diff | wc -l))
	$(eval BRANCH := $(shell git branch | grep '* master' | wc -l))
	@echo $(DIFF)
ifeq ($(DIFF),0)
ifeq ($(BRANCH),1)
	@echo git add doc/source/*
	@echo git commit -m "updating documentation"
	@echo git push origin master
	make -C doc/ html

	@echo git checkout gh-pages
	@echo git pull origin gh-pages
	@echo cp -r doc/build/html sphinxdoc/
	@echo git add sphinxdoc/*
	@echo git commit -m "updating documentation"
	@echo git push origin gh-pages
	@echo git checkout master

else
	@echo "Only run make doc when on master branch!"
endif
else
	@echo "First commit or stash uncommited changes!"
endif

clean:
	find . -name '*.pyc' -delete
	find . -name '__pycache__' -delete
