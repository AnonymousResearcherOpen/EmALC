package connectives;

import com.google.common.collect.Sets;
import concepts.AtomicConcept;
import formula.Formula;
import roles.AtomicRole;

import java.util.Set;

public class Disjoint extends Formula {
    public Disjoint() {
        super();
    }

    public Disjoint(Formula subsumee, Formula subsumer) {
        super(2);
        this.setSubFormulas(subsumee, subsumer);
    }

    public Set<AtomicConcept> get_c_sig() {
        Formula subsumee = this.getSubFormulas().get(0);
        Formula subsumer = this.getSubFormulas().get(1);
        return Sets.union(subsumee.get_c_sig(), subsumer.get_c_sig());
    }

    public Set<AtomicRole> get_r_sig() {
        Formula subsumee = this.getSubFormulas().get(0);
        Formula subsumer = this.getSubFormulas().get(1);
        return Sets.union(subsumee.get_r_sig(), subsumer.get_r_sig());
    }

    @Override
    public String toString() {
        Formula subsumee = this.getSubFormulas().get(0);
        Formula subsumer = this.getSubFormulas().get(1);
        return subsumee + " \u22e2 " + subsumer;
    }
}
