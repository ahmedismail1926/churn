import pickle

with open('artifacts/models/training_results.pkl', 'rb') as f:
    results = pickle.load(f)

print("\nType:", type(results))
print("\nContent:")
if isinstance(results, dict):
    for key, value in results.items():
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if isinstance(value, dict):
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  Value: {value}")
elif isinstance(results, list):
    print(f"List with {len(results)} items")
    for i, item in enumerate(results[:3]):
        print(f"\nItem {i}: {type(item)}")
        print(item)
else:
    print(results)
