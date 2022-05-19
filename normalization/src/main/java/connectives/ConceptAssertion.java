package connectives;

import concepts.AtomicConcept;
import formula.Formula;
import individual.Individual;

import java.util.ArrayList;
import java.util.List;

public class ConceptAssertion extends Formula {
    protected String text = "";
    public ConceptAssertion(AtomicConcept concept, Individual individual){
        List<Formula> f = new ArrayList<>(); f.add(concept); f.add(individual);
        this.setSubFormulas(f);
        text = concept.getText()+"("+individual.getText()+")";
    }
    public String getText() {return this.text;}

    @Override
    public String toString() {
        return this.getText();
    }
}
