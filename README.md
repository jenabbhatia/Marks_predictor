# Marks_predictor

The CBSE Class 12 examination, is taken by Indian high school students at the end of K-12 school education. The scores or grades in this examination form the basis of their entry to the College or University system, for an undergraduate program. At the K-12 level, students appear for examination in five subjects. These five subjects generally include one language; three elective subjects oriented towards Science, Commerce or Humanities; and any elective of their choice as a fifth subject

This challenge is based on real school data of the CBSE Class 12 examination conducted in the year 2013. You are given the grades obtained by students with specific but popular combinations of subjects (and all these students had opted for Mathematics). Their grades in four subjects are known to you. However their grade in Mathematics (i.e, the fifth subject) is hidden.

The records provided to you are the grades obtained by students who had opted for the following combinations of subjects or courses and obtained a passing grade in each subject. The individual subjects in the data are: 
English, Physics, Chemistry, Mathematics, Computer Science, Biology, Physical Education, Economics, Accountancy and Business Studies.

The grades of students in four subjects (other than Mathematics) are provided to you. Can you predict what grade they had obtained in Mathematics?

To help you build a prediction engine,there is a training file, containing the grade points obtained by students with the above subject combinations, in all five subjects.

Prediction engine is implemented using a Decision Tree Classifier. Predict the target value(5th subject marks) by passing the test set to the decision tree predict function, then calculate the accuracy score between predicted value and the actual value. You will get the accuracy score around 33%.
