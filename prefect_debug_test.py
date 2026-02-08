from prefect import flow, task

@task
def hello():
    print("Hello from Prefect task")

@flow
def test_flow():
    hello()

if __name__ == "__main__":
    test_flow()
