#!/usr/bin/env python

def SourceTargetsDoc(a):
    '''The module generates distant annotations for PICO entities using a cominbation of distant supervision and dynamic programming'''
    return a**a

print( SourceTargetsDoc.__doc__ )

import argparse
import collections
import datetime as dt
import difflib
import itertools
import json
import logging
import operator
import os
import random
import re
import sys
import time
import traceback
from asyncore import write
from collections import Counter, defaultdict
from functools import reduce
from itertools import chain
import itertools
from random import shuffle
from typing import Iterator

import numpy as np
import pandas as pd
import scipy
from elasticsearch import Elasticsearch, helpers
from elasticsearch_dsl import Q, Search
from nltk.tokenize import WhitespaceTokenizer
from pylab import *
from snorkel.labeling import PandasLFApplier
from snorkel.labeling.model import LabelModel

from CandGenUtilities.experiment_arguments import *
from CandGenUtilities.labeler_utilities import *
from CandGenUtilities.source_target_mapping import *
from LabelingFunctions.externalmodelLF import ExternalModelLabelingFunction
from LabelingFunctions.heuristicLF import (heurPattern_p_sampsize,
                                           heurPattern_pa, posPattern_i)
from LabelingFunctions.LFutils import (label_abb_and_write, label_heur_and_write, label_lf_partitions,
                                       label_ont_and_write,
                                       label_regex_and_write,
                                       label_umls_and_write, spansToLabels)
from LabelingFunctions.ontologyLF import *
from load_data import loadEBMPICO
from Ontologies.ontologyLoader import *
from Ontologies.ontoUtils import *

################################################################################
# Set seed
################################################################################
seed = 0
seed_everything(seed)
print('The random seed is set to: ', seed)

################################################################################
# Load stopwords (generic negative label LFs)
################################################################################
#sw_lf = loadStopWords()

################################################################################
# Set global variable
################################################################################
#candgen_version = 'v4' # version = {v3, v4, ...}

extract_abbs = False

if_negs = True

################################################################################
# Parse arguments for experiment flow
################################################################################

for candgen_version in ['v4', 'v3']: # version = {v3, v4, ...}

    if candgen_version == 'v3':
        if_stopwords = True
    elif candgen_version == 'v4':
        if_stopwords = False

    for input_file in ['test_ebm_st_corr', 'val_studytype', 'training_ebm', 'test_ebm_anjani', 'test_physio', 'test_ebm']:

        parser = argparse.ArgumentParser()
        parser.add_argument('-level1', type=bool, default=False) # Level 1 = UMLS LF's
        parser.add_argument('-level2', type=bool, default=False) # Level 2 = Non-UMLS LF's
        parser.add_argument('-level3', type=bool, default=False) # Level 3 = Rule based LF's (ReGeX, Heuristics and handcrafted dictionaries)
        parser.add_argument('-level4', type=bool, default=False) # Level 5 = Abbreviation LFs
        parser.add_argument('-levels', type=bool, default=False) # execute data labeling using all levels
        parser.add_argument('-umls_fpath', type=Path, default= 'UMLS/english_subset/umls_preprocessed/umls_tui_pios4_.db')
        parser.add_argument('-ds_fpath', type=Path, default='/mnt/nas2/data/systematicReview/ds_cto_dict' )
        parser.add_argument('-abb_fpath', type=Path, default='/mnt/nas2/data/systematicReview/abbreviations' )
        parser.add_argument('-indir', type=Path, default='/mnt/nas2/data/systematicReview' ) # directory with labeling function sources
        parser.add_argument('-outdir', type=Path, default=f'/mnt/nas2/results/Results/systematicReview/distant_pico/tui_pio_v4/{input_file}_candidate_generation/{candgen_version}' ) # directory path to store the weakly labeled candidates
        parser.add_argument('-stop', type=bool, default=if_stopwords ) # False = Wont have stopword LF, True = Will have stopword LF
        parser.add_argument('-write_cand', type=bool, default=False ) # Should write candidates? True = Yes - Write , False = No - Dont write
        args = parser.parse_args()

        try:

            ##############################################################################################################
            # Load labelling sources
            ############################################################################################################## 
            print('Retrieving UMLS ontology arm (Preprocessing applied)')
            umls_db = f'{args.indir}/{args.umls_fpath}'
            umls_p  = loadUMLSdb(umls_db, entity='P')
            umls_p_lst = [v_i[0] for k,v in umls_p.items() for v_i in v  if '-' not in v_i[-1] ]   

            umls_i = loadUMLSdb(umls_db, entity='I')
            umls_i_lst = [v_i[0] for k,v in umls_i.items() for v_i in v  if '-' not in v_i[-1] ]

            umls_o = loadUMLSdb(umls_db, entity='O')
            umls_o_lst = [v_i[0] for k,v in umls_o.items() for v_i in v  if '-' not in v_i[-1] ]

            umls_s = loadUMLSdb(umls_db, entity='S')
            umls_s_lst = [v_i[0] for k,v in umls_s.items() for v_i in v  if '-' not in v_i[-1] ]

            print('Retrieving non-UMLS Ontologies  (Preprocessing applied)')
            p_DO, p_DO_syn = loadOnt( f'{args.indir}/Ontologies/participant/DOID.csv', delim = ',', term_index = 1, term_syn_index = 2  )
            p_ctd, p_ctd_syn = loadOnt( f'{args.indir}/Ontologies/participant/CTD_diseases.tsv', delim = '\t', term_index = 0, term_syn_index = 7 )
            p_HPO, p_HPO_syn = loadOnt( f'{args.indir}/Ontologies/participant/HP.csv', delim = ',', term_index = 1, term_syn_index = 2  )
            p_age_onto = loadDict(f'{args.indir}/Ontologies/participant/age_ontology.txt')
            p_Gen, p_Gen_syn = loadOnt( f'{args.indir}/Ontologies/participant/GSSO.csv', delim = ',', term_index = 1, term_syn_index = 2  )
            
            i_ctd, i_ctd_syn = loadOnt( f'{args.indir}/Ontologies/intervention/CTD_chemicals.tsv', delim = '\t', term_index = 0, term_syn_index = 7 )
            i_chebi, i_chebi_syn = loadOnt( f'{args.indir}/Ontologies/intervention/CHEBI.csv', delim = ',', term_index = 1, term_syn_index = 2  )
            i_npi, i_npi_syn = loadOnt( f'{args.indir}/Ontologies/intervention/NPI.csv', delim =  ',', term_index = 0, term_syn_index = 7 )
            i_ncco, i_ncco_syn = loadOnt( f'{args.indir}/Ontologies/intervention/NCCO.csv', delim = ',', term_index = 1, term_syn_index = 2  )

            o_oae, o_oae_syn = loadOnt( f'{args.indir}/Ontologies/outcome/OAE.csv', delim=',', term_index=1, term_syn_index=2 )
            o_so, o_so_syn = loadOnt( f'{args.indir}/Ontologies/outcome/SYMP.csv', delim=',', term_index=1, term_syn_index=2 )
            o_cctoo, o_cctoo_syn = loadOnt( f'{args.indir}/Ontologies/outcome/CCTOO.csv', delim=',', term_index=1, term_syn_index=2 )
            o_ontotox, o_ontotox_syn = loadOnt( f'{args.indir}/Ontologies/outcome/ONTOTOX.csv', delim=',', term_index=1, term_syn_index=2 )
            
            s_cto, s_cto_syn = loadOnt( f'{args.indir}/Ontologies/study_type/CTO.csv', delim=',', term_index=1, term_syn_index=2 )
            s_rctont,_ = loadOnt( f'{args.indir}/Ontologies/study_type/RCTONT.csv', delim=',', term_index=1, term_syn_index=2 )
            s_ocre, _ = loadOnt( f'{args.indir}/Ontologies/study_type/OCRE.csv', delim=',', term_index=1, term_syn_index=2 )
            s_ctont, _ = loadOnt( f'{args.indir}/Ontologies/study_type/CTONT.csv', delim=',', term_index=1, term_syn_index=2 )


            print('Retrieving distant supervision dictionaries')
            ds_participant = loadDS(args.ds_fpath, 'participant')
            ds_intervention = loadDS(args.ds_fpath, 'intervention')
            ds_intervention_syn = loadDS(args.ds_fpath, 'intervention_syn')
            ds_outcome = loadDS(args.ds_fpath, 'outcome')

            ######################################  Dictionary Labeling Functions ###############################
            print('Retrieving hand-crafted dictionaries')
            p_genders = loadDict(f'{args.indir}/Ontologies/participant/gender_sexuality.txt')
            i_comparator = loadDict(f'{args.indir}/Ontologies/intervention/comparator_dict.txt')
            o_endpoints = loadDict(f'{args.indir}/Ontologies/outcome/endpoints_dict.txt')
            s_dictionary = loadDict(f'{args.indir}/Ontologies/study_type/rct.txt')

            ###################################### ReGeX Labeling Function ######################################
            print('Retrieving ReGeX patterns')
            p_sampsize = loadPattern( 'samplesize' ) # Generic pattern 
            p_sampsize2 = loadPattern( 'samplesize2' ) # Sample size in ReGeX expression (n=XXXX)
            p_sampsize3 = loadPattern( 'samplesize3' )  # Digits expressed as words 0_999
            p_sampsize4 = loadPattern( 'samplesize4' )  # Digits expressed as words one_to_999
            p_sampsize5 = loadPattern( 'samplesize5' )  # Digits expressed as words one_to_999_999

            p_age = loadPattern( 'age0' )
            p_agerange = loadPattern( 'age1' )
            p_agemax = loadPattern( 'age2' )
            p_agemaxmin = loadPattern( 'age3' )
            p_meanage = loadPattern( 'meanage' )

            i_control = loadPattern( 'control_i' )

            o_adverse = loadPattern( 'adverse_o' )

            s_study_type = loadPattern( 'studytype' )
            s_study_type_basic = loadPattern( 'studytype_basic' )
            s_study_type_basicplus = loadPattern( 'studytype_basic+' )
            s_study_type_proc = loadPattern( 'studytype_procedure' )
            s_study_type_s = loadPattern( 'studytypes_var' )
            s_placebo = loadPattern( 'control_i' )
            s_blinding = loadPattern( 'studytype_binded' )
            s_phase = loadPattern( 'study_phase' )

            # Negative non-UMLS labels
            negative_s_nonUMLS = itertools.chain( p_DO, p_DO_syn, p_ctd, p_ctd_syn, p_HPO, p_HPO_syn, o_oae, o_oae_syn, o_so, o_so_syn, i_ctd, i_ctd_syn, i_chebi, i_chebi_syn)
            negative_s_nonUMLS = list(set(negative_s_nonUMLS))
            negative_s_nonUMLS = [ i for i in negative_s_nonUMLS if i.lower() not in list(map(str.lower, i_comparator))] # controls, placebo, sham, saline, etc...

            # Negatove regex and heuristics
            negative_s_regex = itertools.chain( ds_outcome, ds_participant, p_genders, o_endpoints, p_DO, p_DO_syn, p_ctd, p_ctd_syn, p_HPO, p_HPO_syn, o_oae, o_oae_syn, o_so, o_so_syn, i_ctd, i_ctd_syn, i_chebi, i_chebi_syn)
            negative_s_regex = list(set(negative_s_regex))
            negative_s_regex = [ i for i in negative_s_regex if i.lower() not in list(map(str.lower, i_comparator))] # controls, placebo, sham, saline, etc...

            ##############################################################################################################
            # Load training, validation and test datasets
            ##############################################################################################################
            ebm_nlp = '/mnt/nas2/data/systematicReview/PICO_datasets/EBM_parsed'
            df_data, df_data_flatten = loadEBMPICO( ebm_nlp, args.outdir, candgen_version=candgen_version, write_to_file = True )

            #########################################################################################
            # Level 1 - UMLS LF's
            #########################################################################################
            if args.level1 == True or args.levels == True:
                
                for m in ['direct', 'fuzzy']: # fuzzy = fuzzy bigram match, direct = no fuzzy bigram match
                    outdir_umls = f'{args.outdir}/UMLS/{m}'
                    for entity, umls_d in zip(['S'], [ umls_s ]):
                        label_umls_and_write(outdir_umls, umls_d, df_data, picos=entity, arg_options=args, write= args.write_cand )


            #########################################################################################
            # Level 2 - Non-UMLS LF's (non-UMLS Ontology labeling)
            #########################################################################################
            if args.level2 == True or args.levels == True:

                for m in ['fuzzy', 'direct']:
                    outdir_non_umls = f'{args.outdir}/nonUMLS/{m}'
                    for ontology, ont_name in zip([ s_ctont, s_ocre ,s_rctont, s_cto, s_cto_syn, s_ctont, s_ocre ], ['s_ctont', 's_ocre', 's_rctont', 's_cto', 's_cto_syn', 's_ctont', 's_ocre'] ) :
                        if if_negs == True:
                            nonUMLS_s_labels = label_ont_and_write( outdir_non_umls, ontology, picos='S', df_data=df_data, write=args.write_cand, arg_options=args, ontology_name=ont_name, extra_negs=negative_s_nonUMLS)
                        else:
                            nonUMLS_s_labels = label_ont_and_write( outdir_non_umls, ontology, picos='S', df_data=df_data, write=args.write_cand, arg_options=args, ontology_name=ont_name)

            ##############################################################################################################
            # Level 3 - Rule based LF's (ReGeX, Heuristics and handcrafted dictionaries)
            ##############################################################################################################
            if args.level3 == True or args.levels == True:

                ###################################### Fetch negative labelling function ######################################
                neg_i_dict = itertools.chain( p_genders, o_endpoints )
                neg_i_dict = list(set(neg_i_dict))
                neg_i_dict_filtered = [ i for i in neg_i_dict if i.lower() not in list(map(str.lower, i_comparator))]

                neg_s_dict = itertools.chain( p_genders, o_endpoints )
                neg_s_dict = list(set(neg_s_dict))
                neg_s_dict_filtered = [ i for i in neg_i_dict if i.lower() not in list(map(str.lower, i_comparator))]
                neg_s_dict_filtered = [ i for i in neg_s_dict_filtered if i.lower() not in list(map(str.lower, s_dictionary))]

                # Dictionary Labeling Function
                for m in ['fuzzy', 'direct']:
                    for ontology, entity, ont_name, neg_labs  in zip([s_dictionary, i_comparator], ['S', 'S'], ['dict_s_type', 'dict_s_comp_type'], [ neg_s_dict_filtered, neg_s_dict_filtered ] ) : 
                        outdir_dict = f'{args.outdir}/dictionary/{m}'
                        dict_labels = label_ont_and_write( outdir_dict, ontology, picos=entity, df_data=df_data, write=args.write_cand, arg_options=args, ontology_name=ont_name, extra_negs=neg_labs)

                ######################################  ReGeX Labeling Functions ######################################

                for reg_lf_i, entity, reg_lf_name in zip([s_study_type, s_study_type_basic, s_study_type_basicplus, s_study_type_proc, s_study_type_s, s_placebo, s_blinding, s_phase], ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S'], ['regex_stdtype', 'regex_stdtype_basic', 'regex_stdtype_basicplus', 'regex_stdtype_proc', 'regex_stdtype_types', 'regex_placebo', 'regex_blinding', 'regex_phase' ] ) : 
                    outdir_reg = f'{args.outdir}/heuristics/direct'
                    print('Regex labeling on...')
                    if if_negs == True:
                        label_regex_and_write( outdir_reg, [reg_lf_i], picos=entity, df_data=df_data, write=args.write_cand, arg_options=args, lf_name=reg_lf_name, neg_labs = negative_s_regex )
                    else:
                        label_regex_and_write( outdir_reg, [reg_lf_i], picos=entity, df_data=df_data, write=args.write_cand, arg_options=args, lf_name=reg_lf_name )

                ######################################  Heutistic Labeling Functions ###################################### 
                # TODO: Negative labelling function
                outdir_heurPattern = f'{args.outdir}/heuristics/direct'

                filename = 'lf_' + str('lf_s_heurpattern_labels') + '.tsv'
                if if_negs == True:
                    label_heur_and_write( outdir_heurPattern, picos='S', df_data=df_data, write=args.write_cand, arg_options=args, lf_name=str(filename).replace('.tsv', ''), tune_for='sensitivity', neg_labs=negative_s_regex)
                else:
                    label_heur_and_write( outdir_heurPattern, picos='S', df_data=df_data, write=args.write_cand, arg_options=args, lf_name=str(filename).replace('.tsv', ''))

                filename = 'lf_' + str('lf_s_heurpattern_labels') + '.tsv'
                if if_negs == True:
                    label_heur_and_write( outdir_heurPattern, picos='S', df_data=df_data, write=args.write_cand, arg_options=args, lf_name=str(filename).replace('.tsv', ''), tune_for='specificity', neg_labs=negative_s_regex )
                else:
                    label_heur_and_write( outdir_heurPattern, picos='S', df_data=df_data, write=args.write_cand, arg_options=args, lf_name=str(filename).replace('.tsv', ''), tune_for='specificity' )


                filename = 'lf_' + str('lf_s_heurpattern_labels_2') + '.tsv'
                if if_negs == True:
                    label_heur_and_write( outdir_heurPattern, picos='S', df_data=df_data, write=args.write_cand, arg_options=args, lf_name=str(filename).replace('.tsv', ''), tune_for='sensitivity', neg_labs=negative_s_regex )
                else:
                    label_heur_and_write( outdir_heurPattern, picos='S', df_data=df_data, write=args.write_cand, arg_options=args, lf_name=str(filename).replace('.tsv', ''), tune_for='sensitivity' )

        except Exception as ex:

            template = "An exception of type {0} occurred. Arguments:{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print( message )

            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(traceback.format_exc())

            logging.info(message)
            string2log = str(exc_type) + ' : ' + str(fname) + ' : ' + str(exc_tb.tb_lineno)
            logging.info(string2log)
            logging.info(traceback.format_exc())