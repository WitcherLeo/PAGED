

def get_prompt():
    prompt = """I want you to generate the Procedural Graph based on a Procedural Document.
The Procedural Graph contains the following types of "Nodes" and "Flows":

"Nodes":
"Start": start node indicates the start of a procedure, represented as "Start".
"End": end node indicates the ending of a procedure, represented as "End".
"Action": action node indicates a specific step in a procedure, represented as the step itself, such as "prepare the ingredients".
"XOR": exclusive gateway, indicates that only one of the following non-sequential actions can be executed, distinguish by numbers, such as "XOR1".
"OR": inclusive gateway, indicates that one or more of the following non-sequential actions can be executed, distinguish by numbers, such as "OR1".
"AND": parallel gateway, indicates that all of the following actions should be executed in parallel, distinguish by numbers, such as "AND1".
"DataConstraint": DataConstraint indicates the constraints for the necessary data of the actions, represented as "DataConstraint(data object)".
"ActionConstraint": ActionConstraint indicates essential notices need to be considered for the execution of the actions, represented as "ActionConstraint(essential notices)".

" Flows":
"SequenceFlow": flow that represents the execution of sequential actions, such as "Start -> prepare the ingredients". 
"ConditionFlow": the condition flow is used to indicate that the following action is performed under the condition on the Condition Flow, such as "XOR1 -> (condition1) choose the first one".
"ConstraintFlow": flow that is used to connect the constraints with corresponding actions, such as "prepare the ingredients -> ActionConstraint(essential notices)".

In addition, the actor of corresponding actions is put in the front of corresponding elements to indicate the actor of the following actions if needed, such as "For actor1:".

You should generate the graph in the format of "Node -> Node" line by line until generating the whole graph for the given Procedural Document, and keep the text of the nodes and conditions consistent with the original Procedural Document.

Here are some examples:

###
"Procedural Document":
Firstly, the customer needs to find an empty seat. If the customer needs dishes, then choose the desired dishes and specify the taste. If the customer needs drinks, then order the drinks and specify the size. The customer then submits the order, which is added to the order list. After enjoying the meal, the customer should choose the payment method. If the credit card is available, the customer pays by credit card; else if the credit card is not available, the customer should pay in cash. For the restaurant, once receiving the order from the order list, it prepares the meal according to the order and prepares the tableware for the customer at the same time. The meal is then served for the customer to enjoy. After that, the restaurant asks the customer to pay for the order and then confirms the payment. Note that the restaurant should provide the receipt if the customer needs. And the procedure ends. 

###
"Procedural Graph":
The procedure starts with the customer finding an empty seat. Then there is an inclusive gateway to indicate the non-sequential actions because the customer can need dishes or drinks or both, and the customer should specify the taste  after choosing the desired dishes and specify the size after ordering the drinks. Then the customer should submit the order to produce the order list data and enjoy the meal and then choose payment method. Then there is an exclusive gateway to indicate the non-sequential actions because the credit card is available or not. If the credit card is available, the customer pays by credit card; else if the credit card is not available, the customer should pay in cash. And the procedure for the customer ends. Then for the restaurant, there is a parallel gateway after receiving the order from the order list to indicate the non-sequential actions because the restaurant should prepare the meal and prepare the tableware in parallel. Then the restaurant serves the meal, asks the customer to pay for the order and confirms the payment. And note that provide the receipt if the customer needs when confirming the payment. And the procedure for the restaurant ends.
So the Procedural Graph of this Procedural Document is as follows:

For the customer:
Start -> find an empty seat
find an empty seat -> OR1
OR1 -> (needs dishes) choose the desired dishes
OR1 -> (needs drinks) order the drinks
choose the desired dishes  -> specify the taste
order the drinks -> specify the size
specify the taste -> OR2
specify the size -> OR2
OR2 -> submits the order
submits the order -> DataConstraint(order list)
submits the order -> enjoy the meal
enjoy the meal -> choose payment method
choose payment method -> XOR1
XOR1 -> (credit card is available) pay by credit card
XOR1 -> (credit card is unavailable) pay in cash
pay by credit card -> XOR2
pay in cash -> XOR2
XOR2 -> End

For the restaurant:
Start -> receive an order
DataConstraint(order list) -> receive an order
receive an order -> AND1
AND1 -> prepare the meal
AND1 -> prepare the tableware
prepare the meal -> AND2
prepare the tableware -> AND2
AND2 -> serve the meal
serve the meal -> ask the customer to pay for the order
ask the customer to pay for the order -> confirm the payment
confirm the payment -> ActionConstraint(provide the receipt if the customer needs)
confirm the payment -> End


###
"Procedural Document":
In the beginning, the staff will receive an order request, and then checks the order type. If the order is standard type, the sufficience of the stock is checked according to the stock table. If the order is special type, upload the order to the factory system. If the stock is sufficient for standard order, the goods will be directly shipped out, else if the stock is insufficient, they will need to be transferred from other warehouses. After that, the staff updates the order status and provide order information to the user. At the same time, the staff needs to bind order information to user account. Finally, the staff record the request status and the procedure ends.

###
"Procedural Graph":
The procedure starts with the staff receive an order request. After checking the order type, there is an exclusive gateway to indicate the non-sequential actions because the order can be standard type or special type. If the order is special type, upload the order to the factory system. And if the order is standard type, the sufficience of the stock is checked according to the stock table data. And there is one more exclusive gateway after checking the sufficience of the stock to indicate the non-sequential actions because the stock can be sufficient or insufficient. If the stock is sufficient for standard order, the goods will be directly shipped out, else if the stock is insufficient, they will need to be transferred from other warehouses. Then there is a parallel gateway to indicate the non-sequential actions because the staff should update the order status and provide order information to the user and meanwhile, bind order information to user account. Finally, the staff records the request status and the procedure ends.
So the Procedural Graph of this Procedural Document is as follows:

For the staff:
Start -> receive an order request
receive an order request -> check the order type
check the order type -> XOR1
XOR1 -> (the order is standard type) check the sufficience of the stock
XOR1 -> (the order is special type) upload the order to the factory system
DataConstraint(the stock table) -> check the sufficience of the stock
check the sufficience of the stock -> XOR2
XOR2 -> (the stock is sufficient) directly shipped out the goods
XOR2 -> (the stock is insufficient) transfer the goods from other warehouses
directly shipped out the goods -> XOR3
transfer the goods from other warehouses -> XOR3
XOR3 -> XOR4
upload the order to the factory system -> XOR4
XOR4 -> AND1
AND1 -> update the order status
update the order status -> provide order information to the user
AND1 -> bind order information to user account
provide order information to the user -> AND2
bind order information to user account -> AND2
AND2 -> record the request status
record the request status -> End


###
"Procedural Document":
Start the service by receiving the email from the electronic mailbox, then parse the email content. If the email contains account query request, reply the account information to the user. If the email contains account modification request, record the information needs to be modified. After that, verify the validity of the account and verify the legality of the modified information at the same time if there exists account information to be modified. Otherwise update the verification timestamp of the account directly. Finally, synchronize the email content to the system and the procedure ends.

###
"Procedural Graph":
The procedure starts with receiving the email from the electronic mailbox data. Then parse the email content and there is an inclusive gateway to indicate the non-sequential actions because the email can contain account query request or account modification request or both. If the email contains account query request, reply the account information to the user. If the email contains account modification request, record the information needs to be modified. After that, there is an exclusive gateway to indicate the non-sequential actions because there exists account information to be modified or not. If there exists no account information to be modified, update the verification timestamp of the account directly. Else if there exists account information to be modified, there is a parallel gateway to indicate the non-sequential actions because we should verify the validity of the account and verify the legality of the modified information in parallel. Then synchronize the email content to the system and the procedure ends.
So the Procedural Graph of this Procedural Document is as follows:

Start -> receive the email
DataConstraint(electronic mailbox) -> receive the email
receive the email -> parse the email content
parse the email content -> OR1
OR1 -> (the email contains account query request) reply the account information to the user
OR1 -> (the email contains account modification request) record the information needs to be modified
reply the account information to the user -> OR2
record the information needs to be modified -> OR2
OR2 -> XOR1
XOR1 -> (there exists account information to be modified) AND1
XOR1 -> (otherwise) update the verification timestamp of the account directly
AND1 -> verify the validity of the account
AND1 -> verify the legality of the modified information
verify the validity of the account -> AND2
verify the legality of the modified information -> AND2
AND2 -> XOR2
update the verification timestamp of the account directly -> XOR2
XOR2 -> synchronize the email content to the system
synchronize the email content to the system -> End


Now you need to generate the corresponding Procedural Graph of the following Procedural Document:

###
"Procedural Document":
{}

###
"Procedural Graph":
"""

    return prompt


def get_prompt_for_train():
    prompt = """I want you to generate the Procedural Graph based on a Procedural Document.
The Procedural Graph contains the following types of "Nodes" and "Flows":

"Nodes":
"Start": start node indicates the start of a procedure, represented as "Start".
"End": end node indicates the ending of a procedure, represented as "End".
"Action": action node indicates a specific step in a procedure, represented as the step itself, such as "prepare the ingredients".
"XOR": exclusive gateway, indicates that only one of the following non-sequential actions can be executed, distinguish by numbers, such as "XOR1".
"OR": inclusive gateway, indicates that one or more of the following non-sequential actions can be executed, distinguish by numbers, such as "OR1".
"AND": parallel gateway, indicates that all of the following actions should be executed in parallel, distinguish by numbers, such as "AND1".
"DataConstraint": DataConstraint indicates the constraints for the necessary data of the actions, represented as "DataConstraint(data object)".
"ActionConstraint": ActionConstraint indicates essential notices need to be considered for the execution of the actions, represented as "ActionConstraint(essential notices)".

" Flows":
"SequenceFlow": flow that represents the execution of sequential actions, such as "Start -> prepare the ingredients". 
"ConditionFlow": the condition flow is used to indicate that the following action is performed under the condition on the Condition Flow, such as "XOR1 -> (condition1) choose the first one".
"ConstraintFlow": flow that is used to connect the constraints with corresponding actions, such as "prepare the ingredients -> ActionConstraint(essential notices)".

In addition, the actor of corresponding actions is put in the front of corresponding elements to indicate the actor of the following actions if needed, such as "For actor1:".

You should generate the graph in the format of "Node -> Node" line by line until generating the whole graph for the given Procedural Document, and keep the text of the nodes and conditions consistent with the original Procedural Document.

Here are some examples:

###
"Procedural Document":
Firstly, the customer needs to find an empty seat. If the customer needs dishes, then choose the desired dishes and specify the taste. If the customer needs drinks, then order the drinks and specify the size. The customer then submits the order, which is added to the order list. After enjoying the meal, the customer should choose the payment method. If the credit card is available, the customer pays by credit card; else if the credit card is not available, the customer should pay in cash. For the restaurant, once receiving the order from the order list, it prepares the meal according to the order and prepares the tableware for the customer at the same time. The meal is then served for the customer to enjoy. After that, the restaurant asks the customer to pay for the order and then confirms the payment. Note that the restaurant should provide the receipt if the customer needs. And the procedure ends. 

###
"Procedural Graph":
The procedure starts with the customer finding an empty seat. Then there is an inclusive gateway to indicate the non-sequential actions because the customer can need dishes or drinks or both, and the customer should specify the taste  after choosing the desired dishes and specify the size after ordering the drinks. Then the customer should submit the order to produce the order list data and enjoy the meal and then choose payment method. Then there is an exclusive gateway to indicate the non-sequential actions because the credit card is available or not. If the credit card is available, the customer pays by credit card; else if the credit card is not available, the customer should pay in cash. And the procedure for the customer ends. Then for the restaurant, there is a parallel gateway after receiving the order from the order list to indicate the non-sequential actions because the restaurant should prepare the meal and prepare the tableware in parallel. Then the restaurant serves the meal, asks the customer to pay for the order and confirms the payment. And note that provide the receipt if the customer needs when confirming the payment. And the procedure for the restaurant ends.
So the Procedural Graph of this Procedural Document is as follows:

For the customer:
Start -> find an empty seat
find an empty seat -> OR1
OR1 -> (needs dishes) choose the desired dishes
OR1 -> (needs drinks) order the drinks
choose the desired dishes  -> specify the taste
order the drinks -> specify the size
specify the taste -> OR2
specify the size -> OR2
OR2 -> submits the order
submits the order -> DataConstraint(order list)
submits the order -> enjoy the meal
enjoy the meal -> choose payment method
choose payment method -> XOR1
XOR1 -> (credit card is available) pay by credit card
XOR1 -> (credit card is unavailable) pay in cash
pay by credit card -> XOR2
pay in cash -> XOR2
XOR2 -> End

For the restaurant:
Start -> receive an order
DataConstraint(order list) -> receive an order
receive an order -> AND1
AND1 -> prepare the meal
AND1 -> prepare the tableware
prepare the meal -> AND2
prepare the tableware -> AND2
AND2 -> serve the meal
serve the meal -> ask the customer to pay for the order
ask the customer to pay for the order -> confirm the payment
confirm the payment -> ActionConstraint(provide the receipt if the customer needs)
confirm the payment -> End


###
"Procedural Document":
In the beginning, the staff will receive an order request, and then checks the order type. If the order is standard type, the sufficience of the stock is checked according to the stock table. If the order is special type, upload the order to the factory system. If the stock is sufficient for standard order, the goods will be directly shipped out, else if the stock is insufficient, they will need to be transferred from other warehouses. After that, the staff updates the order status and provide order information to the user. At the same time, the staff needs to bind order information to user account. Finally, the staff record the request status and the procedure ends.

###
"Procedural Graph":
The procedure starts with the staff receive an order request. After checking the order type, there is an exclusive gateway to indicate the non-sequential actions because the order can be standard type or special type. If the order is special type, upload the order to the factory system. And if the order is standard type, the sufficience of the stock is checked according to the stock table data. And there is one more exclusive gateway after checking the sufficience of the stock to indicate the non-sequential actions because the stock can be sufficient or insufficient. If the stock is sufficient for standard order, the goods will be directly shipped out, else if the stock is insufficient, they will need to be transferred from other warehouses. Then there is a parallel gateway to indicate the non-sequential actions because the staff should update the order status and provide order information to the user and meanwhile, bind order information to user account. Finally, the staff records the request status and the procedure ends.
So the Procedural Graph of this Procedural Document is as follows:

For the staff:
Start -> receive an order request
receive an order request -> check the order type
check the order type -> XOR1
XOR1 -> (the order is standard type) check the sufficience of the stock
XOR1 -> (the order is special type) upload the order to the factory system
DataConstraint(the stock table) -> check the sufficience of the stock
check the sufficience of the stock -> XOR2
XOR2 -> (the stock is sufficient) directly shipped out the goods
XOR2 -> (the stock is insufficient) transfer the goods from other warehouses
directly shipped out the goods -> XOR3
transfer the goods from other warehouses -> XOR3
XOR3 -> XOR4
upload the order to the factory system -> XOR4
XOR4 -> AND1
AND1 -> update the order status
update the order status -> provide order information to the user
AND1 -> bind order information to user account
provide order information to the user -> AND2
bind order information to user account -> AND2
AND2 -> record the request status
record the request status -> End


###
"Procedural Document":
Start the service by receiving the email from the electronic mailbox, then parse the email content. If the email contains account query request, reply the account information to the user. If the email contains account modification request, record the information needs to be modified. After that, verify the validity of the account and verify the legality of the modified information at the same time if there exists account information to be modified. Otherwise update the verification timestamp of the account directly. Finally, synchronize the email content to the system and the procedure ends.

###
"Procedural Graph":
The procedure starts with receiving the email from the electronic mailbox data. Then parse the email content and there is an inclusive gateway to indicate the non-sequential actions because the email can contain account query request or account modification request or both. If the email contains account query request, reply the account information to the user. If the email contains account modification request, record the information needs to be modified. After that, there is an exclusive gateway to indicate the non-sequential actions because there exists account information to be modified or not. If there exists no account information to be modified, update the verification timestamp of the account directly. Else if there exists account information to be modified, there is a parallel gateway to indicate the non-sequential actions because we should verify the validity of the account and verify the legality of the modified information in parallel. Then synchronize the email content to the system and the procedure ends.
So the Procedural Graph of this Procedural Document is as follows:

Start -> receive the email
DataConstraint(electronic mailbox) -> receive the email
receive the email -> parse the email content
parse the email content -> OR1
OR1 -> (the email contains account query request) reply the account information to the user
OR1 -> (the email contains account modification request) record the information needs to be modified
reply the account information to the user -> OR2
record the information needs to be modified -> OR2
OR2 -> XOR1
XOR1 -> (there exists account information to be modified) AND1
XOR1 -> (otherwise) update the verification timestamp of the account directly
AND1 -> verify the validity of the account
AND1 -> verify the legality of the modified information
verify the validity of the account -> AND2
verify the legality of the modified information -> AND2
AND2 -> XOR2
update the verification timestamp of the account directly -> XOR2
XOR2 -> synchronize the email content to the system
synchronize the email content to the system -> End


Now you need to generate the corresponding Procedural Graph of the following Procedural Document:

###
"Procedural Document":
{}

###
"Procedural Graph":
{}"""


    return prompt

