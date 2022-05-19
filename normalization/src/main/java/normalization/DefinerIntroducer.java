package normalization;

import concepts.AtomicConcept;

import java.util.ArrayList;
import java.util.List;

public class DefinerIntroducer {
    public List<AtomicConcept> definers;
    public DefinerIntroducer(){
        definers = new ArrayList<>();
    }
    public AtomicConcept introduceDefiner(){
        int id = definers.size()+1;
        AtomicConcept c = new AtomicConcept("definer_"+id);
        c.setDefiner_index(id);
        definers.add(c);
        return c;
    }
}
