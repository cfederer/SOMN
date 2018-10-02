This code will reproduce results from Federer & Zylberberg A self-organizing short-term dynamical memory network. https://www.ncbi.nlm.nih.gov/pubmed/30007123 

Each file has the arguments set to run the results as shown in the paper, i.e. to re-create the results from Fig. 2a-c run:

python fig2a_dynamics.py
and
python fig2bc_update_rules.py 

To run the results in the supplemental section for networks with initially stronger synaptic strengths, set args['g'] = 1.6 (also works with various strengths but 1.6 was used in the paper results)

Please e-mail calliefederer@gmail.com with any errors or questions. 
