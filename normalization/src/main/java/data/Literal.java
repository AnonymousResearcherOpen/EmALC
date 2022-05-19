package data;

import concepts.AtomicConcept;
import formula.Formula;
import org.apache.commons.lang3.builder.HashCodeBuilder;

import java.util.LinkedHashSet;

public class Literal extends AtomicConcept {

    public Literal() {
        super();
    }

    public Literal(String str) {
        super(str);
    }


    @Override
    public String toString() {
        return this.getText();
    }

    @Override
    public int hashCode() {
        return new HashCodeBuilder(17, 37).append(this.getText()).toHashCode();
    }

}
