package semantics;

import concepts.AtomicConcept;
import connectives.ConceptAssertion;
import formula.Formula;
import individual.Individual;
import javafx.util.Pair;
import roles.AtomicRole;

import java.util.*;

public class MapFunction {

    protected Map<AtomicConcept, Set<Individual>> c2i;
    protected Map<Individual, Set<AtomicConcept>> i2c;
    protected Map<AtomicRole, Set<Pair<Individual, Individual>>> r2i;
    protected Set<Individual> iSet;
    protected Set<AtomicConcept> cSet;
    protected int size;
    public int size_c; public int size_r;

    public Map<AtomicConcept, Set<Individual>> get_c2i() {return this.c2i;}
    public Map<Individual, Set<AtomicConcept>> get_i2c() {return this.i2c;}
    public Map<AtomicRole, Set<Pair<Individual, Individual>>> get_r2i() {return this.r2i;}
    public Set<Individual> get_iSet() {return this.iSet;}

    public int getSize() {return this.size_c+this.size_r;}

    public Set<Individual> getRoleDomain(AtomicRole role){
        Set<Individual> domain = new HashSet<>();
        for(Pair<Individual,Individual> pair:this.r2i.get(role)){
            domain.add(pair.getKey());
        }
        return domain;
    }
    public Set<Individual> getSpecificRoleRange(AtomicRole role, Set<Individual> i){
        Set<Individual> range = new HashSet<>();
        for(Pair<Individual,Individual> pair:this.r2i.get(role)){
            if(i.contains(pair.getKey()))
                range.add(pair.getValue());
        }
        return range;
    }
    public Set<Individual> getRoleRange(AtomicRole role){
        Set<Individual> range = new HashSet<>();
        for(Pair<Individual,Individual> pair:this.r2i.get(role)){
            range.add(pair.getValue());
        }
        return range;
    }

    public Set<Individual> getNegation(AtomicConcept concept){
        Set<Individual> set = new HashSet<>();
        set.addAll(iSet);
        set.removeAll(c2i.get(concept));
        return set;
    }
    public MapFunction(List<Formula> formulaList, Set<AtomicConcept> cSet){
        c2i = new HashMap<>(); i2c = new HashMap<>(); r2i = new HashMap<>(); iSet = new HashSet<>();
        addAssertions(formulaList);
        this.cSet = cSet;
        size = 0;
        size_c=0;size_r = 0;
    }

    public void addAssertions(List<Formula> formulaList){
        for(Formula f:formulaList) addAssertion(f);
    }

    public void addAssertion(Formula f){
        if(f instanceof ConceptAssertion){
            AtomicConcept c = (AtomicConcept) f.getSubFormulas().get(0);
            if(c.neg){ // ToDo: other strategy
                Individual i = (Individual) f.getSubFormulas().get(1);
                for(AtomicConcept nc:cSet){
                    if(nc.equals(c)) continue;
                    if(i2c.containsKey(i) && i2c.get(i).contains(nc)) return;
                    c2i.putIfAbsent(nc,new HashSet<>()); i2c.putIfAbsent(i, new HashSet<>());
                    c2i.get(nc).add(i);
                    i2c.get(i).add(nc);
                    iSet.add(i);
                    size_c+=1;
                }

            }else{
                Individual i = (Individual) f.getSubFormulas().get(1);
                if(i2c.containsKey(i) && i2c.get(i).contains(c)) return;
                c2i.putIfAbsent(c,new HashSet<>()); i2c.putIfAbsent(i, new HashSet<>());
                c2i.get(c).add(i);
                i2c.get(i).add(c);
                iSet.add(i);
                size_c+=1;
            }

        }else{
            AtomicRole r = (AtomicRole) f.getSubFormulas().get(0);
            Individual i1 = (Individual) f.getSubFormulas().get(1);
            Individual i2 = (Individual) f.getSubFormulas().get(2);
            Pair<Individual, Individual> pair = new Pair<Individual, Individual>(i1,i2);
            if(r2i.containsKey(r) && r2i.get(r).contains(pair)) return;
            r2i.putIfAbsent(r,new HashSet<>());
            r2i.get(r).add(pair);
            iSet.add(i1);
            iSet.add(i2);
            size_r+=1;
        }

    }

    public void removeDisjoint(Formula a, Formula b){
        AtomicConcept left = (AtomicConcept)a;
        AtomicConcept right = (AtomicConcept)b;
        Set<Individual> intersection_i = new HashSet<>();
        intersection_i.addAll(c2i.get(left));
        intersection_i.retainAll(c2i.get(right));

        for(Individual i:intersection_i){
            if(i.getText().equals(left.getText()+"_individual_0")) {
                i2c.get(i).remove(right);
                c2i.get(right).remove(i);
            }
            else if(i.getText().equals(right.getText()+"_individual_0")){
                i2c.get(i).remove(left);
                c2i.get(left).remove(i);
            }
            else{
                i2c.get(i).remove(right);i2c.get(i).remove(left);
                c2i.get(right).remove(i);c2i.get(left).remove(i);
            }
        }
    }



}
