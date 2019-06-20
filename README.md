# Automatic Collation of Parallel Texts

Author: Lucas van der Deijl, University of Amsterdam  
Version: 20 June 2019

## Aim of this program

The aim of this program is to automate three tasks: 
+ to align equivalent sentences from a pair of parallel texts*; 
+ to formalise and calculate the relative difference between these equivalent sentences;
+ to extract the lexical differences per aligned sentence-pair.

*A pair of parallel texts can consist of two different (mono-lingual) translations of a given source text or of two variants of the same text (a manuscript and its printed equivalent, two different editions of the same text etc.)

## Pipeline

The pipeline desigend to achieve the program's aim performs the following steps:
+ Import the required libraries
+ Install the required libraries (if needed)
+ Load the files
+ Set the parameters
+ Preprocess the texts
+ Create a list of unique words
+ Create chunks ('windows') of a set number of consecutive sentences from both texts
+ Align equivalent windows and sentences
+ Calculate similarity between aligned sentence pairs
+ Visualise the results

## How to use

This program can either be run as a separate python-script or a Jupyter Notebook. If you prefer to use Jupyter, the results of the analysis will be both visualised directly and stored as .png images in the directory of the notebook. In addition, a copy of all results will be stored in a structured .txt-file.

The Jupyter notebook can be used to inspect the code or to replicate the collation-task with your own pair of parallel texts. Run each code block individually or use the 'Run all'-option from the Cell-tab. The two input files should be stored in the directory of this Jupyter notebook and named 'source.txt' and 'target.txt' (encoded in UTF-8). Two sample files are available in this repository, containing two different Dutch editions of Descartes's _Discours de la méthode_ (1637).

## Requirements
+ Python 3.4 or higher (to run the .py-script)
+ Anaconda + Jupyter (to run the Notebook-version)

N.B. Those who are not interested in running the code can use the static html-version of the Notebook available in this repository.
