# The Majority Vote Paradigm Shift: When Popular Meets Optimal

The code is written in Python 3 and is based on the use of Jupyter.

Install the required packages using:

```
pip install -r requirements.txt
```

Download the required datasets:

```
python3 download_data.py
```

To run the code to obtain all subfigures of Figure 2 and Figure 3 from the paper:

Run all `computations.ipynb`

To run experiments on synthetic data and obtain results as in Table 2:

```
python3 syntethic_exps.py
```

To run experiments on real data and obtain results as in Table 2:

```
python3 real_exps.py
```

To run experiments to obtain Figure 4 (from the main) and Figure 2 (from the Appendix):

```
python3 check_bound_looseness.py
```

To run experiments to obtain Table 1 from the Appendix:

```
python3 new_ideas.py
```

To run experiments to confirm Section 3.4 from the main paper (different reliability):

```
python3 multiple_reliability.py
```

To run experiments to confirm Section 3.4 from the main paper (two annotator classes):

```
python3 two_annotator_classes.py
```

## Citation
If you use this code in your research or project, please cite us:
```bibtex
@article{purificato2025majority,
  title={The Majority Vote Paradigm Shift: When Popular Meets Optimal},
  author={Purificato, Antonio and Bucarelli, Maria Sofia and Nelakanti, Anil Kumar and Bacciu, Andrea and Silvestri, Fabrizio and Mantrach, Amin},
  journal={arXiv preprint arXiv:2502.12581},
  year={2025}
}
```
For doubts or errors feel free to ping purificato@diag.uniroma1.it!

## Acknowledgments

The implementation of competitor methods draws from the [Toloka library](https://github.com/Toloka/toloka-kit) and the paper [A Lightweight, Effective, and Efficient Model for LabelAggregation in Crowdsourcing](https://github.com/yyang318/LA_onepass). We gratefully acknowledge the authors for making their code available.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the LICENSE NAME HERE License.