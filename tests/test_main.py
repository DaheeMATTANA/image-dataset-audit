from image_dataset_audit import main


def test_main_prints_greeting(capsys):
    main()
    assert capsys.readouterr().out == "Hello from image-dataset-audit!\n"
