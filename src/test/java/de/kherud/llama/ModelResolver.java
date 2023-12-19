package de.kherud.llama;

import java.nio.file.Paths;


/**
 * An enum which enables us to resolve the model home from system parameters and full model paths.
 */
public enum ModelResolver {
  MODEL_HOME("model.home", "Please pass the system property \"%s\" to the test. "
      + "This should represent the location on local disk where your models are located. "
      + "If you are running this via maven, please run with a -Dmodel.home=/path/to/model/dir. "
      + "Make sure that the directory that you pass exists." ),
  INTEGRATION_TEST_MODEL_NAME("integration.test.model", "The system property \"%s\" is not set.  If you are running this from an IDE, please set it.  If you are running this from Maven, this should be set automatically and there is something strange going on." );
  final String systemPropertyName;
  final String errorMessage;
  ModelResolver(String systemPropertyName, String errorMessage) {
    this.systemPropertyName = systemPropertyName;
    this.errorMessage = errorMessage;
  }

  public String resolve() {
    String ret = System.getProperty(systemPropertyName);
    if(ret == null) {
      throw new IllegalArgumentException(String.format(errorMessage, systemPropertyName));
    }
    return ret;
  }

  public static String getPathToModel(String modelName) {
    return Paths.get(MODEL_HOME.resolve(), modelName).toString();
  }
  public static String getPathToITModel() {
    return getPathToModel(INTEGRATION_TEST_MODEL_NAME.resolve());
  }
}
