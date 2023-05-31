import logging
import subprocess


def prepare_database(db_name):
    logging.info("Start PostgreSQL")
    subprocess.run(
        [
            "sudo",
            "--user=postgres",
            "--group=postgres",
            "/etc/init.d/postgresql",
            "start",
        ],
        capture_output=True,
        check=True,
    )
    logging.info("Change PostgreSQL settings to allow trust authentication")
    pg_hba_conf_path = (
        subprocess.run(
            [
                "sudo",
                "--user=postgres",
                "--group=postgres",
                "--login",
                "--",
                "psql",
                "--command=show hba_file;",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        .stdout.splitlines()[2]
        .strip()
    )
    pg_hba_conf_content = "local all all trust\n"
    with open(pg_hba_conf_path, "w", encoding="utf-8") as f:
        f.write(pg_hba_conf_content)
    logging.info("Reload PostgreSQL")
    subprocess.run(
        [
            "sudo",
            "--user=postgres",
            "--group=postgres",
            "/etc/init.d/postgresql",
            "reload",
        ],
        capture_output=True,
        check=True,
    )
    logging.info(f"Create database {db_name}")
    subprocess.run(
        ["createdb", "--username", "postgres", db_name],
        capture_output=True,
        check=True,
    )


def close_database():
    logging.info("Stop PostgreSQL")
    subprocess.run(
        [
            "sudo",
            "--user=postgres",
            "--group=postgres",
            "/etc/init.d/postgresql",
            "stop",
        ],
        capture_output=True,
        check=True,
    )
