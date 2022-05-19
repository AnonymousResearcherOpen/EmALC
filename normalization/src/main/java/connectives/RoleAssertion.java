package connectives;

import formula.Formula;
import individual.Individual;
import roles.AtomicRole;


import java.util.ArrayList;
import java.util.List;

public class RoleAssertion extends Formula {
    protected String text = "";
    public RoleAssertion(AtomicRole role, Individual individual1, Individual individual2){
        List<Formula> f = new ArrayList<>(); f.add(role); f.add(individual1); f.add(individual2);
        this.setSubFormulas(f);
        text = role.getText()+"("+individual1.getText()+", "+individual2.getText()+")";
    }
    public String getText() {return this.text;}

    @Override
    public String toString() {
        return this.getText();
    }
}
