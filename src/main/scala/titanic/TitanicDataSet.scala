package titanic

import titanic.NaiveBayes.{calcPriorPropabilities, countAttributeValues, findBestFittingClass, getAttributeValues, modelwithAddOneSmoothing}

object TitanicDataSet {

  /**
   * Creates a model that predicts 1 (survived) if the person of the certain record
   * is female and 0 (deceased) otherwise
   *
   * @return The model represented as a function
   */
  def simpleModel:(Map[String, Any], String) => (Any, Any)= {
    (map: Map[String, Any], String ) => if(map("sex") == "male") ("PassengerId", 0) else ("PassengerId", 1)
  }

  /**
   * This function should count for a given attribute list, how often an attribute is
   * not present in the data records of the data set
   *
   * @param data    The DataSet where the counting takes place
   * @param attList List of attributes where the missings should be counted
   * @return A Map that contains the attribute names (key) and the number of missings (value)
   */
  def countAllMissingValues(data: List[Map[String, Any]], attList: List[String]): Map[String, Int] = {
    data.flatten.filter(x => attList.contains(x._1)).groupBy(x => (x._1)).mapValues(data.size - _.size).filter(x => x._2 != 0)
  }

  /**
   * This function should extract a set of given attributes from a record
   *
   * @param record  Record that should be extracted
   * @param attList List of attributes that should be extracted
   * @return A Map that contains only the attributes that should be extracted
   *
   */
  def extractTrainingAttributes(record:Map[String, Any], attList:List[String]):Map[String, Any]= {
    record.filter(x => attList.contains(x._1))
  }

  /**
   * This function should create the training data set. It extracts the necessary attributes,
   * categorize them and deals with the missing values. You could find some hints in the description
   * and the lectures
   *
   * @param data Training Data Set that needs to be prepared
   * @return Prepared Data Set for using it with Naive Bayes
   */
  def createDataSetForTraining(data:List[Map[String, Any]]): List[Map[String, Any]] = {

    val dataattr = data.map(x => extractTrainingAttributes(x , List("sex","age","pclass", "survived")))

    val agemean = (dataattr.map(x => x.filter(f => f._1 == "age")).flatMap(x => x).map(x => x._2.asInstanceOf[Float]).sum)/(dataattr.map(x => x.filter(f => f._1 == "age")).size)

    def categorize(age: Any): Float = {
      val x = age.toString.toFloat
      if (x <= 4) 1 else if (x <= 15) 2 else if (x <= 50) 3 else 4
    }

    dataattr.map(x => if(x.contains("age")) x else x + ("age" ->agemean)).map(y => y.updated("age", categorize(y("age"))))
  }

  /**
   * This function builds the model. It is represented as a function that maps a data record
   * and the name of the id-attribute to the value of the id attribute and the predicted class
   * (similar to the model building process in the train example)
   *
   * @param trainDataSet  Training Data Set
   * @param classAttrib name of the attribute that contains the class
   * @return A tuple consisting of the id (first element) and the predicted class (second element)
   */
  def createModelWithTitanicTrainingData(tdata:List[Map[String,Any]], classAttr:String): (Map[String, Any], String) => (Any, Any)= {

    val d = createDataSetForTraining(tdata)
    val classVals= countAttributeValues(d,classAttr)
    val aValues = getAttributeValues(d).asInstanceOf[ Map[String, Set[Any]]]
    val prior= calcPriorPropabilities(d,classAttr)
    val data: Map[Any, Set[(String, Map[Any, Int])]] = NaiveBayes.calcAttribValuesForEachClass(d,classAttr)
    val condProp = data.transform((z, y) => y.map(x => (x._1, aValues(x._1).map(t => t -> ((x._2.getOrElse(t, 0).toDouble + 1.0) /(classVals(z.toString.toInt) + aValues(x._1).size ))).toMap)))

    (map,id_key) => (map(id_key),findBestFittingClass(NaiveBayes.calcClassValuesForPrediction(map-id_key,condProp,prior)))

  }
}
