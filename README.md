# DF-ALC

This repository provides the code for training DF-ALC, a neural-symbolic model that can use the knowledge with expressivity no more than $\mathcal{ALC}$ from any OWL ontologies to guide the learning of neural models.

## Overview

Input of DF-ALC is an OWL ontology and a neural model. In the structure of DF-ALC, the domain of discourse is the object set of neural model, and the signature is the union of concept names and role names in the ontology. Through reconstructing the output of neural model into an interpretation of the ontology, we can revise the neural model through learn with the hierarchical loss presented in the paper. The parameters of the neural model can also be revised in a way of multi-task learning, which is easy to be extended with the codes in this repository.
Output of DF-ALC is the revised neural model output (/revised neural model parameters with the extension).

## Details for Reproducing

### Preprocessing for the OWL ontology

The input OWL ontology should be truncated into its $\mathcal{ALC}$ fragment, and then be normalized. 
Run the following command with **JDK 1.8** under [the root of this directory](https://github.com/AnonymousResearcherOpen/DF-ALC/): 

`java -jar Normalization.jar training/ontologies training/input`

The output of preprocessing is the files in 'training/input':

- 'concepts.txt', 'roles.txt', 'individuals.txt': the concept names(/role names/individual names) set.
- 'normalization.txt': the nomalized TBox axioms.
- 'abox.txt': the abox assertions.
- 'subclassaixoms.txt': the GCI axioms.

Note: The source code of 'Normalization.jar' and 'CQGenerator.jar' is in [normalization](https://github.com/AnonymousResearcherOpen/DF-ALC/tree/main/normalization).If you want to repackage the jar based on our source code, remember to delete all dependencies named as 'owlapi-xxx.jar' in the artifact, while only remain the 'owlapi-distribution-5.1.3.jar'. 

### Training

The training and evaluation is in [training](https://github.com/AnonymousResearcherOpen/DF-ALC/tree/main/training), to train DF-ALC, run:
`python .\run.py --info_path input --out_path output --save_path output --iter_path ontologies --mask_rate 0.2 --alpha 0.8 --device_name cpu`

For evaluation, we randomly masked the ABox of the input ontology as the initial output of the neural models, so can evaluate the performance of DF-ALC when meeting with different distributions. The generation of the masked ABox (imitation of the output of a neural model) is in [Evaluation.MaskABox](https://github.com/AnonymousResearcherOpen/DF-ALC/tree/main/training/Evaluation.py), the masked ABox and the origional ABox are saved in '--save_path'. And the mask rate is designated by '--mask_rate'. While '--alpha' is the threshold of truth value for the transformation between fuzzy ALC and crisp ALC. And the masked value is in the range of (1-alpha,alpha). The model also support using GPU, with '--device_name cuda:0'.

For comparison with the Logical Tensor Network, run:
`python .\run.py --info_path input --out_path output --save_path output --iter_path ontologies --mask_rate 0.2 --alpha 0.8 --device_name cpu --model_name LTN`


### Evaluation

The revised results of D-ALC and LTN are evaluated under the semantics of fuzzy first order logic, with codes in [training/evaluation](https://github.com/AnonymousResearcherOpen/DF-ALC/tree/main/training/evaluation/). To compute the successful rate, run [run.ipynb](https://github.com/AnonymousResearcherOpen/DF-ALC/tree/main/training/evaluation/run.ipynb)

To do the conjuncive query answering (CQA) evaluation, firstly,
generate the conjunctive queries and answers:

`java -jar CQGenerator.jar training/ontologies training/input`

Then run [CQAnswering_evaluation.ipynb](https://github.com/AnonymousResearcherOpen/DF-ALC/tree/main/training/CQAnswering_evaluation.ipynb) to generate the CQA evaluation results.

## Dependencies
> JDK 1.8
> python 3.7.0
> torch 1.8.1
> python-csv 0.0.13
> matplotlib 3.3.2
> pickle 4.0
> numpy 1.21.4
> pandas 1.1.3
> pyparsing 3.0.6
> loguru 0.6.0
## Results
Results of DF-ALC and LTN are output in [output](https://github.com/AnonymousResearcherOpen/DF-ALC/tree/main/training/output/), [product_output](https://github.com/AnonymousResearcherOpen/DF-ALC/tree/main/training/product_output/), respectively. We zipped the training results in [results](https://drive.google.com/drive/folders/1ob0RVM6GwAQvgew9yZTrCfNrfvbWFKRb?usp=sharing).


