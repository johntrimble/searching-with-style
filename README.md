# Searching with Style

[Notebook](code/similarity.ipynb) demonstrating how to do an image search based on style similarity using [Cassandra Vector Search](https://www.datastax.com/blog/introducing-vector-search-empowering-cassandra-astra-db-developers-to-build-generative-ai-applications).


## Setup

Download the [WikiArt](https://archive.org/details/wikiart-dataset) dataset and extract it into the `wikiart` directory at the root of the repository (not the one under the `code` directory).

Download the [Best Artworks of All Time](https://www.kaggle.com/datasets/ikarus777/best-artworks-of-all-time) (not my name for the dataset, I'm not an art critic) and extract it into the `best_artwork` directory at the root of the repository (again, not the one under the `code` directory).

Create a Vector Search Database using "style_search" as the keyspace (instructions [here](https://docs.datastax.com/en/astra-serverless/docs/vector-search/create-astra-vector-database.html)).

Download the secure connect bundle for your database and place it in the root of the repository.

Create a `.env` file in the root of the repository with the following contents:

```shell
ASTRA_CLIENT_ID=YOUR_CLIENT_ID
ASTRA_CLIENT_SECRET=YOUR_CLIENT_SECRET
ASTRA_SCB_FILENAME=NAME_OF_SECURE_CONNECT_BUNDLE.zip
```

## Running the notebook

You have two options for running the notebook:

1. Open the repository in VS Code and select "Reopen in Container" from the Command Palette.
2. Run `docker compose up --build` directly from the command line and access the notebook at http://localhost:8888
