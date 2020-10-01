# pyMG 

This Python code is a more or less structured framework for linear multigrid solvers. It contains the sources, 
the exercises and the slides for the lecture "Einführung in Mehrgitter-Methoden" at FH Aachen, spring semester 2016. 
More details can be found in the slides and the documentation.


## Structure of the project

    .
    |\
    | bin                scripts implementing specific example runs
    |\
    | docs
    |  \
    |   source           RestDoc files picked up by Sphinx to generate documentation
    |\
    | project            implementations by the user (read: student) using pyMG
    |\
    | pymg               core functionality of pyMG
     \
      test
      |\
      | project_tests    tests for the user's implementations
      \
        pymg_tests       tests for the core functionality


## Notes for Developers

### Run Test Suite

We use [nose][] for our tests.
Simply follow the general guides on Python `unittest` and *nosetest*.


### Update and Build Documentation of the code 

We use [Sphinx][sphinx] with it's bundled [autosummary][] and [autodoc][] extensions as the
documentation processor.

As soon as a new file, class or module is created either in `pymg` or `project`, it is required to
rerun *autodoc* to pick up the new stuff.
To do this run this handy script we wrote in the root of the project:

    ./docs/update_apidocs.sh

To just rebuild the documentation, change into the `docs` folder and run

    make html


### Contributing

Read [CONTRIBUTING](CONTRIBUTING).


## License

In short: 2-clause BSD.
See [LICENSE](LICENSE).

[nose]: https://nose.readthedocs.org/en/latest/
[sphinx]: http://www.sphinx-doc.org/en/stable/index.html
[autosummary]: http://www.sphinx-doc.org/en/stable/ext/autosummary.html
[autodoc]: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
