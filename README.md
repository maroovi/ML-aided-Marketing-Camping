
# ML Assisted Marketing Camping

Targeting a potential customer using various mediums like social media, telemarketing, 
digital media and paid partnerships to lure in the customers. But how can banks predict 
with accuracy to a specific group of customers using a variety of parameters like 
location, marital status, type of campaign etc. This is where predictive analytics 
comes in to spot light, where we use the data and machine learning models to strategize 
insights from the fields to layout which customers are likely to opt in for a product 
offered by the bank. Predictive analytics favor banks in a greater scale to find which 
demographic of customers are the golden goose here.


## Appendix

- Use Case
- Author
- Feature
- Roadmap
- Running Test
- Lesson
- Acknowledgements

## Author

- [@vignesh](https://github.com/maroovi)


## Features

    21 variables:
    $ age : int 56 57 37 40 56 45 59 41 24 25 ...
    $ job : int 56 57 37 40 56 45 59 41 24 25 ...
    $ marital : chr "housemaid" "services" "services" "admin." ... : chr "married" "married" "married" "married" ...
    $ education : : chr "basic.4y" "high.school" "high.school" "basic.6y" ... : chr "no" "unknown" "no" "no" ...
    $ default :  chr "no" "no" "yes" "no" ... : chr "no" "no" "no" "no" ...
    $ contact : chr "telephone" "telephone" "telephone" "telephone" ... : chr "may" "may" "may" "may" ...
    $ housing : chr "no" "no" "yes" "no" ...
    $ loan : chr "no" "no" "no" "no" ...
    $ month : chr "may", "may", "may"...
    $ day_of_week : chr "mon" "mon" "mon" "mon" ...
    $ duration : int 261 169 339 678 ...
    $ campaign : int 1 1 1 1 1 ...
    $ pdays : int 999 999 999 999 999 ...
    $ previous : int 0 0 0 0 0 0 0 0 ....
    $ poutcome : 
    $ emp.var.rate : num 1.1 1.1 1.1 1.1 1.1 1.1 1.1 1.1 1.1 1.1 ...
    $ cons.price.idx: num 94 94 94 94 94 ...
    $ cons.conf.idx : num -36.4 -36.4 -36.4 -36.4 -36.4 -36.4 -36.4 -36.4 -36.4 -36.4 ... $ euribor3m : num 4.86 4.86 4.86 4.86 4.86 ...
    $ nr.employed : num 5191 5191 5191 5191 5191 ...
    $ y : chr "no" "no" "no" "no" ...
<img src="https://user-images.githubusercontent.com/60556766/150849785-df9dc390-7637-4e0e-aa6a-e6244259cef7.png">
 
## Roadmap

- traditionally we follow the same data mining approach towards this campign

 <img src="https://user-images.githubusercontent.com/60556766/150848755-75afd4d8-3ed4-4a85-885b-b8ebb73ab83c.png">


## Running Tests 

<img width="750" alt="Screen Shot 2022-01-24 at 2 05 02 PM" src="https://user-images.githubusercontent.com/60556766/150850004-dba3c00d-4e15-4e23-8f16-4ad53d3bc7a5.png">
<img width="750" alt="Screen Shot 2022-01-24 at 2 05 39 PM" src="https://user-images.githubusercontent.com/60556766/150850008-e7fa36d8-77d7-48dc-b894-9c646f5c2d80.png">
<img width="500" alt="Screen Shot 2022-01-24 at 2 05 59 PM" src="https://user-images.githubusercontent.com/60556766/150850010-43800632-53f2-43ed-a0af-abf10c16e1e6.png">

## Lessons Learned

We will start with multiple models using PCA data and the original data. We have 
identified that the Logistic Regression model with the PCA data has better performance 
compared to the other models that we have here. However, given the condition of having 
58 features is highly debatable, the ideal solution would be to understand the features 
importance using an expert opinion, which adds values to the machine recommendation.

We have also done with few model tuning methods that shall improve the performance of 
the models that we had built, consider the Logistic Regression, where we employed for 
the Stepwise Regression approach to choose the features of the significance rather than 
including all the features.

## Acknowledgements

 - [Dataflair data mining steps](https://data-flair.training/blogs/wp-content/uploads/sites/2/2019/04/data-mining-steps.jpg)



