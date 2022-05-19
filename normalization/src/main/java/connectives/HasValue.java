package connectives;

import com.google.common.collect.Sets;
import concepts.AtomicConcept;
import concepts.ConceptExpression;
import data.DataProperty;
import data.Literal;
import formula.Formula;
import roles.AtomicRole;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

public class HasValue extends ConceptExpression {
    public HasValue() {
        super();
    }

    public HasValue(DataProperty role, Literal filler) {
        super(role, filler);
//        System.out.println(role+" "+filler);
		this.c_sig = new HashSet<>();
		this.r_sig = new HashSet<>();
		this.set_c_sig(filler.get_c_sig());
		this.set_r_sig(role.get_r_sig());
		this.set_r_sig(filler.get_r_sig());
    }

    public Set<AtomicConcept> get_c_sig() {
        Formula filler = this.getSubFormulas().get(1);
        return filler.get_c_sig();
//        return Collections.emptySet();
    }

    public Set<AtomicRole> get_r_sig() {
//        return Collections.emptySet();
        Formula role = this.getSubFormulas().get(0);
        Formula filler = this.getSubFormulas().get(1);
        return Sets.union(role.get_r_sig(), filler.get_r_sig());
    }

    @Override
    public String toString() {
        Formula role = this.getSubFormulas().get(0);
        Formula filler = this.getSubFormulas().get(1);

        return ""+role+"("+filler+")";
    }

}
