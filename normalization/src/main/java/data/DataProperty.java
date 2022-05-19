package data;

import formula.Formula;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import roles.AtomicRole;

import java.util.LinkedHashSet;
import java.util.List;

public class DataProperty  extends AtomicRole {
    public DataProperty() {
        super();
    }

    public DataProperty(String str) {
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
