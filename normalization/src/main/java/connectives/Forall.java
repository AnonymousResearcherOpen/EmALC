/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package connectives;

import com.google.common.collect.Sets;
import concepts.AtomicConcept;
import concepts.ConceptExpression;
import formula.Formula;
import roles.AtomicRole;

import java.util.Set;


public class Forall extends ConceptExpression {

	public Forall() {
		super();
	}

	public Forall(Formula role, Formula filler) {
		super(role, filler);
	}

	public Set<AtomicConcept> get_c_sig() {
		Formula filler = this.getSubFormulas().get(1);
		return filler.get_c_sig();
	}

	public Set<AtomicRole> get_r_sig() {
		Formula role = this.getSubFormulas().get(0);
		Formula filler = this.getSubFormulas().get(1);
		return Sets.union(role.get_r_sig(), filler.get_r_sig());
	}

	@Override
	public String toString() {
		Formula role = this.getSubFormulas().get(0);
		Formula filler = this.getSubFormulas().get(1);

		if (filler instanceof And || filler instanceof Or) {
			return "\u2200" + role + ".(" + filler + ")";
		} else {
			return "\u2200" + role + "." + filler;
		}
	}
}
