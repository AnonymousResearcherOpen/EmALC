import pyparsing as pp

class AxiomParser(object):
    def __init__(self):
        # ConceptName = pp.Combine(pp.Suppress("<") + pp.Word(pp.printables, excludeChars="()<>{}") + pp.Suppress(">")) ^ pp.Combine(pp.Word(pp.alphanums) + ':' + pp.Word(pp.alphanums))
        ConceptName = pp.Combine(pp.Suppress("<") + pp.Word(pp.printables, excludeChars="()<>{}") + pp.Suppress(">")) ^ pp.Literal("owl:Thing") ^ pp.Literal("owl:Nothing")

        RoleName = ConceptName

        ClassExpression = pp.Forward()

        rule_1 = ConceptName
        rule_2 = pp.Group("ObjectComplementOf"   + pp.Suppress("(") + ClassExpression + pp.Suppress(")"))
        rule_3 = pp.Group("ObjectIntersectionOf" + pp.Suppress("(") + pp.OneOrMore(ClassExpression) +  pp.Suppress(")"))
        rule_4 = pp.Group("ObjectUnionOf"        + pp.Suppress("(") + pp.OneOrMore(ClassExpression) +  pp.Suppress(")"))
        rule_5 = pp.Group("ObjectSomeValuesFrom" + pp.Suppress("(") + RoleName + ClassExpression +  pp.Suppress(")"))
        rule_6 = pp.Group("ObjectAllValuesFrom" + pp.Suppress("(") + RoleName + ClassExpression +  pp.Suppress(")"))

        ClassExpression <<= (rule_1 | rule_2 | rule_3 | rule_4 | rule_5 | rule_6)

        self.__ClassExpression = ClassExpression
        self.__SubClassOfAxiom = pp.Suppress("SubClassOf(") + ClassExpression + ClassExpression + pp.Suppress(")")

    def parse_subclassof_axiom(self, axiom_str):
        return self.__SubClassOfAxiom.parseString(axiom_str).asList()

    def parse_class_expression(self, class_expression_str):
        return self.__ClassExpression.parseString(class_expression_str).asList()

if __name__ == '__main__':
    parser = AxiomParser()
    print(parser.parse_subclassof_axiom(
        r"SubClassOf(owl:Nothing owl:Thing)"
    ))
