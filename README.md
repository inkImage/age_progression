# Age progression

This repo contains some code related to my master's work dedicated to age progression.
There's no finished application. There are just some pieces of code, data, 3rdparty libraries and sources of docs. So it wasn't enough to perform actual age progression but it was enough to get a master's degree :)
The docs are in russian, english abstract is provided below.

# Abstract

Age detection and age progression (or regression) problems are among the most interesting problems in computer vision area.

Age detection problem has a lot of useful applications such as age-dependent user interfaces, audience analysis and research in marketing area, image search systems and search of people based on images.

Age progression (or regression) lets the user to imitate the changes in age on specified photo. There are a lot of applications of this problem such as missing people search (especially children), automatic updates of photos in photo bases of employees, age-invariant face detection and the most obvious application - aesthethical age correction.

The changes in face appearance when aging depends on so many factors that even a human isn't able to detect age by photo accurately enough. The creation of the algorithm that would be able to solve this problem is much more complicated due to the fact that we don't have exhaustive knowledge on how to transform face features we detect to the age of the subject on a photo.

Automatic age imitation should go after prior face analysis that aims to extract changes in face appearance between ages. To get rendered image it's needed to preprocess the source image, then detect current age and finally apply the changes between the source age and target age.

The targets of this work are the following.
1. Perform a research on currently exising methods of photo-based age detection and age imitation.
2. Perform an analysis of these methods, define what sub-tasks should be solved and...
3. ...implement the most important ones of them.
Since age imitation problem includes age detection problems it's recommended to consider age detection problem as a part of age imitation system.

