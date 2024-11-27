from collections import defaultdict, deque

class Cost:
    def __init__(self) -> None:
        self.maxi = 0
        self.sum = 0

def dfs(adj, current, node, color, vis):
    if current == node:
        color[current] = 1
        vis[current] = 1
        return True
    
    ans = False
    for nd in adj[current]:
        if dfs(adj, nd, node, color, vis):
            ans = True
    
    if ans and not vis[current]:
        color[current] = min(2, color[current] + 1)
        vis[current] = 1

    return ans

def create_subgraph(n, adj, color):
    new_adj = defaultdict(list)
    for i in adj.keys():
        for nd in adj[i]:
            if color[i] == 2 and color[nd] == 2:
                new_adj[i].append(nd)

    return new_adj

def nodes_to_consider(color):
    return {i for i, c in color.items() if c == 2}

def lca(u, v, adj, parent):
    n = len(adj)
    vis = {i: 0 for i in adj.keys()}
    color = {i: 0 for i in adj.keys()}

    start = None
    for nd, val in parent.items():
        if len(val) == 0: start = nd; break

    dfs(adj, start, u, color, vis)
    vis = {i: 0 for i in adj.keys()}
    dfs(adj, start, v, color, vis)

    nodes = nodes_to_consider(color)
    adj_sub = create_subgraph(n, adj, color)

    for i in adj.keys():
        if i in nodes and len(adj_sub[i]) == 0:
            return i

    return 0

def lca_multiple(nodes, adj, parent):
    if not nodes:
        return -1
    if len(nodes) == 1:
        return nodes[0]
    current_lca = nodes[0]
    for node in nodes[1:]:
        current_lca = lca(current_lca, node, adj, parent)

    return current_lca

def compute_cost(n, adj, parent, indegree_a, cost):
    total_costs = 0
    indegree = indegree_a.copy()
    costs = {i: Cost() for i in adj.keys()}
    q = deque()

    for i in adj.keys():
        if indegree[i] == 0:
            total_costs += cost[i]
            q.append(i)

    while q:
        node = q.popleft()

        if indegree[node] == 0:
            tmp = cost[node]
            ss = 0
            mm = 0
            for x in parent[node]:
                ss += costs[x].sum
                mm = max(mm, costs[x].sum)

            tmp += (ss + mm) / 2.0
            costs[node].maxi = max(costs[node].maxi, tmp)
            costs[node].sum += tmp

            nodes = parent[node]
            lc = lca_multiple(nodes, adj, parent)
            if lc != -1 and len(parent[node]) == 1 and len(adj[lc]) == 1:
                total_costs += cost[node]

        for nd in adj[node]:
            indegree[nd] -= 1
            if indegree[nd] == 0:
                if indegree_a[nd] > 1:
                    nodes = parent[nd]
                    lc = lca_multiple(nodes, adj, parent)
                    for x in nodes:
                        costs[x].maxi = 0
                        costs[x].sum -= (costs[lc].sum + costs[lc].maxi) / 2.0
                q.append(nd)

    for i in adj.keys():
        if indegree_a[i] > 1:
            total_costs += (costs[i].maxi + costs[i].sum) / 2.0

    return total_costs

def solve():
    n, m = map(int, input().split())
    cost = list(map(int, input().split()))

    indegree = {i: 0 for i in range(n)}
    adj = defaultdict(list)
    parent = defaultdict(list)

    for _ in range(m):
        u, v = map(int, input().split())
        adj[u].append(v)
        parent[v].append(u)
        indegree[v] += 1

    total_cost = compute_cost(n, adj, parent, indegree, cost)
    print(total_cost)


if __name__ == "__main__":
    solve()
