# Smart Grid Load Balancing Using Particle Swarm Optimization

A Python simulation that allocates electricity demand across generators while
respecting capacity limits and minimizing operating cost.

## Run the interactive web application

```bash
pip install -r requirements.txt
streamlit run app.py
```

The browser page lets the user enter generator names, capacities, costs,
electricity demand, and PSO settings manually. Click **Optimize Load
Allocation** to view the best safe allocation and its charts.

## Run in terminal mode (optional)

```bash
python main.py
```

## Project flow

1. Read manually entered generator capacities, costs, and demand-zone data from `app.py`.
2. Create a swarm of random, feasible generator load allocations.
3. Calculate a fitness score for every allocation.
4. Update each particle using its personal best and the swarm's global best.
5. Repair each allocation so generation exactly equals demand and never exceeds capacity.
6. Print the best allocation and save performance charts.

## Fitness function

`Fitness = generation cost + demand mismatch penalty + overload penalty`

The penalties are deliberately large, so supplying the required demand safely
is more important than finding a small cost reduction.

## Educational note

This is a college-project simulation using fictional values. It is not designed
to control a real electrical grid.
