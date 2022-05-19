from functools import reduce
import numpy as np
from loguru import logger
import pickle
from axiom_parser import AxiomParser
from computer import EmbeddingComputer

logger.remove(0)

class EmbeddingEvaluator(object):
    def __init__(self, axiom_path, cname_pickle, rname_pickle):
        self.tbox_axioms = np.loadtxt(axiom_path, dtype=str, delimiter='\n', comments=None)

        with open(cname_pickle, 'rb') as f:
            self.c2id = pickle.load(f)

        with open(rname_pickle, 'rb') as f:
            self.r2id = pickle.load(f)

        assert isinstance(self.c2id, dict)
        assert isinstance(self.r2id, dict)

        self.n_concepts = len(set(self.c2id.values()))
        self.n_roles = len(set(self.r2id.values()))

        logger.info("Load {} concept names with {} unique", len(self.c2id), len(set(self.c2id.values())))
        logger.info("Load {} role names with {} unique", len(self.r2id), len(set(self.r2id.values())))

        self.parser = AxiomParser()

        self.concept_embedding = None
        self.role_embedding = None

    def load_embedding(self, cEmb_path, rEmb_path):

        self.concept_embedding = np.load(cEmb_path, allow_pickle=True)
        self.role_embedding = np.load(rEmb_path, allow_pickle=True)

        logger.info("Load concept embedding of shape {}".format(self.concept_embedding.shape))
        logger.info("Load role embedding of shape {}".format(self.role_embedding.shape))

        # assert self.concept_embedding.shape[0] == self.n_concepts
        # assert self.role_embedding.shape[0] == self.n_roles, "{} != {}".format(self.role_embedding.shape[0], self.n_roles)

        self.concept_embedding = np.clip(self.concept_embedding, 0.0, 1.0)
        self.role_embedding = np.clip(self.role_embedding, 0.0, 1.0)

        assert self.concept_embedding.max() <= 1
        assert self.concept_embedding.min() >= 0, str(self.concept_embedding.min())
        if self.role_embedding.shape[0] > 0:
            assert self.role_embedding.max() <= 1, str(self.role_embedding.max())
            assert self.role_embedding.min() >= 0

        self.n_individuals = self.concept_embedding.shape[1]

    def get_role_embed_by_name(self, role_name:str):
        return self.role_embedding[self.r2id[role_name]]

    def get_concept_embed_by_name(self, concept_name:str):
        return self.concept_embedding[self.c2id[concept_name]]

    def __compute_cembed(self, operator, args):
        # print("operator: {}".format(operator), "args: {}".format(args))

        if operator == "ObjectComplementOf":
            assert len(args) == 1
            x = self.get_concept_embed_by_parsed_class_expr(args[0])
            return self.computer.negate(x)
        elif operator == "ObjectIntersectionOf":
            assert len(args) >= 2
            embeds = [self.get_concept_embed_by_parsed_class_expr(arg) for arg in args]
            return reduce(self.computer.intersect, embeds)
        elif operator == "ObjectUnionOf":
            assert len(args) >= 2
            embeds = [self.get_concept_embed_by_parsed_class_expr(arg) for arg in args]
            return reduce(self.computer.unify, embeds)
        elif operator == "ObjectSomeValuesFrom":
            assert len(args) == 2
            r = self.get_role_embed_by_name(args[0])
            x = self.get_concept_embed_by_parsed_class_expr(args[1])
            return self.computer.exists(r, x)
        elif operator == "ObjectAllValuesFrom":
            assert len(args) == 2
            r = self.get_role_embed_by_name(args[0])
            x = self.get_concept_embed_by_parsed_class_expr(args[1])
            return self.computer.forall(r, x)
        else:
            logger.warning("Unimplemented operator: {}".format(operator))
            return np.zeros(shape=(self.n_individuals,))

    def get_concept_embed_by_parsed_class_expr(self, parsed_class_expr):
        identifier = str(parsed_class_expr)
        embed = self.class_expression_identifer_to_embedding.get(identifier)
        if embed is not None: return embed

        if isinstance(parsed_class_expr, str):
            embed = self.concept_embedding[self.c2id[parsed_class_expr]]
        else:
            operator = parsed_class_expr[0]
            args = parsed_class_expr[1:]
            embed = self.__compute_cembed(operator, args)

        self.class_expression_identifer_to_embedding[identifier] = embed
        return embed

    def evaluate(self, mode):
        if self.concept_embedding is None or self.role_embedding is None:
            assert 0, "Embedding not initialized"
        assert mode in ['godel', 'product', 'crisp']
        self.computer = EmbeddingComputer(mode)

        if mode == 'crisp':
            threshold = 0.5
            self.concept_embedding[self.concept_embedding > 0.5] = 1
            self.concept_embedding[self.concept_embedding <= 0.5] = 0
            self.role_embedding[self.role_embedding > 0.5] = 1
            self.role_embedding[self.role_embedding <= 0.5] = 0
            logger.info("Converting embedding to boolean values by threshold {}", threshold)

        self.class_expression_identifer_to_embedding = {}
        truth_degree_list = []
        parsed_axioms = []  
        for axiom_str in self.tbox_axioms:
            try:
                parsed_subclass, parsed_superclass = self.parser.parse_subclassof_axiom(axiom_str)
            except Exception as e:
                # logger.error("failed to parse {}", axiom_str[:20])
                continue

            parsed_axioms.append(axiom_str)
            
            subclass_embed = self.get_concept_embed_by_parsed_class_expr(parsed_subclass)
            # print(subclass_embed)
            superclass_embed = self.get_concept_embed_by_parsed_class_expr(parsed_superclass)

            assert subclass_embed.shape == superclass_embed.shape

            truth_degree = self.computer.compute_subclassof_truth_degree(subclass_embed, superclass_embed)

            truth_degree_list.append(truth_degree)

        logger.info("Evaluating {} TBox axioms, {} parsed", len(self.tbox_axioms), len(truth_degree_list))
        logger.info("Average: {}", np.average(truth_degree_list))
        return {
            'all_axiomss' : len(self.tbox_axioms),
            'n_parsed_axioms': len(parsed_axioms),
            'parsed_axioms': parsed_axioms,
            'score': np.average(truth_degree_list),
            'detail': truth_degree_list,
        }


if __name__ == '__main__':
    base_name = 'Family2.owl'
    axiom_path = 'to_parse/' + base_name + '.txt'
    cname_pickle = base_name + '.c2id.pkl'
    rname_pickle = base_name + '.r2id.pkl'
    evaluator = EmbeddingEvaluator(axiom_path, cname_pickle, rname_pickle)
    cEmb_path = base_name + '_true_cEmb.npy'
    rEmb_path = base_name + '_true_rEmb.npy'
    evaluator.load_embedding(cEmb_path, rEmb_path)
    evaluator.evaluate(mode='godel')


        