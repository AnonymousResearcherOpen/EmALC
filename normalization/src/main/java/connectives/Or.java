/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package connectives;

import concepts.AtomicConcept;
import concepts.ConceptExpression;
import roles.AtomicRole;
import roles.RoleExpression;
import individual.Individual;
import formula.Formula;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;


public class Or extends Formula {

	public Or() {
		super();
	}
	
	public Or(Set<Formula> set) {
		super(set.size());
		this.setSubformulae(set);
		this.setSubFormulas(new ArrayList<>(this.getSubformulae()));
	}

	public static  Formula getOr(Set<Formula> set){
		Formula or = null;
		if(set.size() > 1) or = new Or(set);
		else or = set.iterator().next();
		return or;
	}

	public Set<AtomicConcept> get_c_sig() {
		Set<Formula> conjunct_set = this.getSubformulae();
		Set<AtomicConcept> ac_set = new LinkedHashSet<>();
		for (Formula conjunct : conjunct_set) {
			ac_set.addAll(conjunct.get_c_sig());
		}
		return ac_set;
	}

	public Set<AtomicRole> get_r_sig() {
		Set<Formula> conjunct_set = this.getSubformulae();
		Set<AtomicRole> ar_set = new LinkedHashSet<>();
		for (Formula conjunct : conjunct_set) {
			ar_set.addAll(conjunct.get_r_sig());
		}
		return ar_set;
	}


	@Override
	public String toString() {
		if (this.getSubFormulas().size() == 1) {
			return this.getSubFormulas().get(0).toString();
		}
		String str = "";
		for (int i = 0; i < this.getSubFormulas().size(); i++) {
			if (i == 0) {
				if (this.getSubFormulas().get(i) instanceof ConceptExpression
						|| this.getSubFormulas().get(i) instanceof RoleExpression
						|| this.getSubFormulas().get(i) instanceof Individual
						|| this.getSubFormulas().get(i) instanceof Negation
						|| this.getSubFormulas().get(i) instanceof Exists
						|| this.getSubFormulas().get(i) instanceof Forall) {
					str = str + this.getSubFormulas().get(i);
					continue;
				}
				str = str + "(" + this.getSubFormulas().get(i) + ")";
				continue;
			}
			if (this.getSubFormulas().get(i) instanceof ConceptExpression
					|| this.getSubFormulas().get(i) instanceof RoleExpression
					|| this.getSubFormulas().get(i) instanceof Individual
					|| this.getSubFormulas().get(i) instanceof Negation
					|| this.getSubFormulas().get(i) instanceof Exists
					|| this.getSubFormulas().get(i) instanceof Forall) {
				str = str + " \u2294 " + this.getSubFormulas().get(i);
				continue;
			}
			str = str + " \u2294 " + "(" + this.getSubFormulas().get(i) + ")";
		}
		return str + "";
	}
}
